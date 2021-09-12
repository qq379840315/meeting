from contextlib import contextmanager
from distutils.version import LooseVersion
import pdb
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types
import logging

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.condition_encoder import ConditionalModule
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        encoder_con: ConditionalModule,
        encoder_rec: AbsEncoder,
        ctc: CTC,
        use_inter_ctc: bool = False,
        use_stop_sign_bce: bool = False,
        inter_ctc_weight: float = 0.3,
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.encoder_con = encoder_con
        self.encoder_rec = encoder_rec
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.use_inter_ctc = use_inter_ctc
        self.adim = encoder.output_size()
        if self.use_inter_ctc:
            self.inter_ctc_weight = inter_ctc_weight
            self.project_linear = torch.nn.Linear(self.encoder_con.output_size(), self.adim)
        self.use_stop_sign_bce = use_stop_sign_bce

        if self.use_stop_sign_bce:
            self.stop_sign_loss = StopBCELoss(self.adim, 1, nunits=self.adim)
        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )
        else:
            self.error_calculator = None
    def min_ctc_loss_and_perm(self, hs_pad, hs_len, ys_pad, text_lengths_new):
        """E2E min ctc loss and permutation.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor hs_len: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, num_spkrs', Lmax)
        :rtype: torch.Tensor
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: minimum index
        """
        
        _, n_left_spkrs, _ = ys_pad.size()
        loss_stack = torch.stack(
            [self.ctc(hs_pad, hs_len, ys_pad[:, i], text_lengths_new[:, i]) for i in range(n_left_spkrs)]
        )  # (N, B, 1)
        min_loss, min_idx = torch.min(loss_stack, 0)
        return min_loss, min_idx
    def resort_sequence(self, xs_pad, min_idx, start):
        """E2E re-sort sequence according to min_idx.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, num_spkrs, Lmax)
        :param torch.Tensor min_idx: batch of min idx (B)
        :param int start: current head of sequence.
        :rtype: torch.Tensor
        :return: re-sorted sequence
        """
        n_batch = xs_pad.size(0)
        for i in range(n_batch):
            tmp = xs_pad[i, start].clone()
            xs_pad[i, start] = xs_pad[i, min_idx[i]]
            xs_pad[i, min_idx[i]] = tmp
        return xs_pad

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        text_all = []
        text_lengths_new = []
        text_lengths_max = int(text_lengths.max())
        for i in range(batch_size):
            text_utt_list=[]
            text_start = 0
            text_end = text_lengths_max
            for j in range(text_lengths_max):
                if text[i][j] == 2:
                    text_utt_list.append(text[i][text_start:j])
                    text_lengths_new.append(j-text_start)
                    text_start = j+1
                elif text[i][j] == -1:
                    text_end = j
                    break
            if(text_start != text_lengths_max):
                text_utt_list.append(text[i][text_start:text_end])
                text_lengths_new.append(text_end-text_start)
            text_all.append(text_utt_list)
        text_lengths_new = torch.Tensor(text_lengths_new).int()
        text_lengths_max = int(text_lengths_new.max())
        text_lengths_new = text_lengths_new.view(batch_size,-1).to(speech.device)
        num_spkrs = text_lengths_new.size(1)
        text_all_final=[]

        for i in range(batch_size):
            text_sequence = pad_sequence(text_all[i],batch_first=True, padding_value=-1)
            pad_num = text_lengths_max - text_sequence.size(-1)
            if pad_num > 0:
                text_sequence = F.pad(text_sequence, (0,pad_num), "constant", -1)
            text_all_final.append(text_sequence)
        text_final_label = torch.stack(text_all_final,dim=0).to(speech.device)
        # for data-parallel
        #text = text[:, : text_lengths.max()]
        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        # encoder_out batchsize * time * dim256
        # encoder_out_lens batchsize
        
        cer_ctc = None

        prev_states = None
        hs_pad_sd, loss_ctc, loss_inter_ctc, loss_stop = (
            [None] * num_spkrs,
            [None] * num_spkrs,
            [None] * num_spkrs,
            [None] * num_spkrs,
        )

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        # encoder_out bc * T * D(256)
        align_ctc_state = encoder_out.new_zeros(encoder_out.size())
        
        for i in range(num_spkrs):
            condition_out, prev_states = self.encoder_con(
                encoder_out, align_ctc_state, encoder_out_lens, prev_states
            )
            # condition_out 8*211*1024
            hs_pad_sd[i], encoder_out_lens,_= self.encoder_rec.forward_hidden(condition_out, encoder_out_lens)
             # hs_pad_sd[i] bc * T * D(256)
            loss_ctc[i], min_idx = self.min_ctc_loss_and_perm(
                hs_pad_sd[i], encoder_out_lens, text_final_label[:, i:], text_lengths_new[:, i:]
            )
            min_idx = min_idx + i
            if i < num_spkrs - 1:
                text_final_label = self.resort_sequence(text_final_label, min_idx, i)
            if self.use_inter_ctc:
                hidden_feature = self.encoder_rec.hidden_feature
                loss_inter_ctc[i] = self.ctc(hidden_feature, encoder_out_lens, text_final_label[:, i], text_lengths_new[:, i])
                logging.info("using latent representation as soft conditions.")
                align_ctc_state = hs_pad_sd[i].detach().data
            if self.use_stop_sign_bce:
                stop_label = hs_pad_sd[i].new_zeros((batch_size, 1))
                if i == num_spkrs - 1:
                    stop_label += 1
                loss_stop[i] = self.stop_sign_loss(hs_pad_sd[i], encoder_out_lens, stop_label)
        loss_ctc = torch.stack(loss_ctc, dim=0).mean()  # (num_spkrs, B)
        loss_stop = torch.stack(loss_stop, dim=0).mean()
        logging.info("ctc loss:" + str(float(loss_ctc)))
        if self.use_inter_ctc:
            loss_inter_ctc = torch.stack(loss_inter_ctc, dim=0).mean()  # (num_spkrs, B)
            logging.info("inter ctc loss:" + str(float(loss_inter_ctc)))
            loss = self.inter_ctc_weight * loss_inter_ctc + (1 - self.inter_ctc_weight) * loss_ctc
            loss_att = loss_inter_ctc
        else:
            loss = loss_ctc
            loss_att = None
        if self.use_stop_sign_bce:
            loss += loss_stop * 100
            acc_att = loss_stop
        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            acc=acc_att.detach() if acc_att is not None else None,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
        )
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
    
        loss_att, acc_att, cer_att, wer_att =  None, None, None, None
        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_rnnt_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        raise NotImplementedError


class StopBCELoss(torch.nn.Module):
    def __init__(
        self, idim, odim=1, nlayers=1, nunits=512, dropout=0, bidirectional=False
    ):
        super(StopBCELoss, self).__init__()
        self.idim = idim
        self.lstmlayers = torch.nn.LSTM(
            idim, nunits, nlayers, batch_first=True, bidirectional=bidirectional
        )
        self.output = torch.nn.Linear(idim, odim)
        self.dropout = torch.nn.Dropout(dropout)

        self.loss = torch.nn.BCELoss()

    def forward(self, xs_pad, xs_len, ys):
        """
        :param torch.Tensor xs_pad: input sequence (B, Tmax, dim)
        :param list xs_len: the lengths of xs (B)
        :param torch.Tensor ys: the labels (B, 1)
        """
        xs_pack = torch.nn.utils.rnn.pack_padded_sequence(
            xs_pad, xs_len.cpu(), batch_first=True
        )
        _, (h_n, _) = self.lstmlayers(xs_pack)  # (B, dim)
        linear_out = self.dropout(self.output(h_n[-1]))  # (B, 1)
        linear_out = torch.sigmoid(linear_out)
        return self.loss(linear_out, ys)
    def get_stop_sign(self, xs_pad, xs_len):
        """
        :param torch.Tensor xs_pad: input sequence (B, Tmax, dim)
        :param list xs_len: the lengths of xs (B)
        :param torch.Tensor ys: the labels (B, 1)
        """
        xs_pack = torch.nn.utils.rnn.pack_padded_sequence(
            xs_pad, xs_len.cpu(), batch_first=True
        )
        _, (h_n, _) = self.lstmlayers(xs_pack)  # (B, dim)
        linear_out = self.dropout(self.output(h_n[-1]))  # (B, 1)
        linear_out = torch.sigmoid(linear_out)
        return linear_out
