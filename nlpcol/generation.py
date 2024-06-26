from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Size, Tensor

if TYPE_CHECKING:
    from nlpcol.models.base import BaseConfig as Config
    from nlpcol.models.base import CausalLMOutput
    from nlpcol.models.t5 import Seq2SeqLMOutput



class GenerationMixin:
    """generation混入类"""
    config: Config

    def init_decoder_input_ids(self, input_ids: torch.Tensor):
        """初始化decoder_input_ids"""
        pass

    def prepare_inputs_for_generation(self):
        pass

    def _update_model_kwargs_for_generation(self, model_kwargs:dict) -> dict:
        """更新model_kwargs字典，为下一个step做准备"""

    def rm_prompt_token_ids(self, decoder_input_ids: torch.LongTensor, input_ids: torch.LongTensor):
        """decode_only模型去掉prompt部分"""
        return decoder_input_ids.tolist()
                
    @torch.inference_mode()
    def generate(
        self, 
        input_ids: torch.LongTensor, # (batch_size, seq_len)
        mode: str = 'greedy_search', # 解码方式
        **model_kwargs
    ) -> List[List[int]]:
        self.eval() # eval模式
        """batch_generate
        解码过程的两阶段：
            prefill stage： 推理过程的初始化步骤，生成 kv_cache 
            decode stage：利用 kv_cache 进行解码，并向 kv_cache 添加新的信息。
        """
        assert mode in ('greedy_search', 'beam_search', 'do_sample')

        if mode == "beam_search":
            return self.generate_beam(input_ids, **model_kwargs)

        batch_size = input_ids.shape[0]
        device = input_ids.device 
        decoder_input_ids = self.init_decoder_input_ids(input_ids)
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        model_kwargs['decoder_input_ids'] = decoder_input_ids

        for step in range(model_kwargs['max_new_tokens']):
            model_inputs:dict = self.prepare_inputs_for_generation(step, input_ids, **model_kwargs)
            outputs:CausalLMOutput = self(**model_inputs)    # 获取dec端的输出

            next_token_logits = outputs.lm_logits[:, -1, :] # 去最后一步的预测结果 (bs, vocab_size)
            next_tokens = getattr(self, mode)(next_token_logits, **model_kwargs) # (bs)

            # 已经完成的序列，next_tokens强制设置为pad_token_id
            if self.config.pad_token_id is not None:
                # e.g. [8467, 8467] * [1, 1] + 0 * (1 - [1, 1])
                next_tokens = next_tokens * unfinished_sequences + self.config.pad_token_id * (1 - unfinished_sequences)

            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim=-1) # next_tokens.unsqueeze(1)
            model_kwargs = self._update_model_kwargs_for_generation(outputs, decoder_input_ids, model_kwargs)

            # 当一个序列遇到eos_token_id时，设置为结束 
            # 这里已经完成的序列依然会继续进行计算，所以存在计算资源浪费的情况，kv_cache下，每次只计算一个token，所以影响并不大
            # e.g. sum(tensor([8467, 8467]) != i for i in [1]) --> tensor([1, 1])
            if self.config.eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((sum(next_tokens != i for i in [self.config.eos_token_id])).long())

                # 当一个batch内每条数据都遇到eos_token_id时，推理结束
                if unfinished_sequences.max() == 0:
                    break
        
        decoder_input_ids = self.rm_prompt_token_ids(decoder_input_ids, input_ids)
        return decoder_input_ids


    def greedy_search(self, scores: Tensor, **model_kwargs) -> Tensor:
        next_tokens = torch.argmax(scores, dim=-1) # (batch_size)
        return next_tokens


    def do_sample(
        self,
        scores: Tensor, # (batch_size, vocab_size)
        top_k: int = None, # 取概率最大的 K 个词
        top_p: float = None,  # 小于1。累积概率超过概率 p 的最小单词集中进行 一般取0.9左右
        temperature: float = 1, # 温度参数(0,2)，temperature<1时，拉大分值间的差异，softmax后分值大的采样概率更高
        **model_kwargs
    ) -> Tensor:
        # top-p 和 top-K 采样于传统的 贪心 和 波束 搜索相比，能产生更流畅的文本
        scores = scores / temperature
        
        if top_k is not None:
            top_k = min(top_k, scores.shape[-1]) # 安全检查
            v = torch.topk(scores, top_k)[0] # (batch_size, top_k) topk值
            top_k_v = v[:, [-1]] # (batch_size, 1) 
            scores[scores < top_k_v] = -float('Inf')  # 小于topk概率的设置-Inf
        
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1) # 先做归一化再cumsum， 归一化也可以用： probs / probs.sum(dim=-1, keepdim=True)
            sorted_indices_to_remove:Tensor = cumulative_probs <= (1 - top_p)

            # 恢复原始概率值的顺序
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            scores[indices_to_remove] = -float('Inf')

        probs = torch.softmax(scores, dim=-1) # 归一化
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze() # (batch_size)
        return next_tokens


    def generate_beam(
        self,
        input_ids: torch.LongTensor, # (batch_size, seq_len)
        num_beams:int=1,
        length_penalty=1.0,
        do_early_stopping=False,
        **model_kwargs
    ) -> Tensor:

        batch_size = input_ids.shape[0]
        device = input_ids.device
        batch_beam_size = batch_size * num_beams

        # 对input_ids张量进行扩展
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)  # (batch_size*num_beams, seq_len)  [0 0 1 1 2 2]
        decoder_input_ids = self.init_decoder_input_ids(input_ids)
        model_kwargs['decoder_input_ids'] = decoder_input_ids

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=input_ids.device,
            length_penalty=length_penalty,
            do_early_stopping=do_early_stopping,
        )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,)) # (batch_size * num_beams)

        for step in range(model_kwargs['max_new_tokens']):
            model_inputs:dict = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            outputs = self(**model_inputs, start_pos = step)    # 获取dec端的输出

            next_token_logits:torch.Tensor = outputs.lm_logits[:, -1, :]
            next_token_scores = next_token_logits.log_softmax(dim=-1)  # (batch_size * num_beams, vocab_size)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )  # (batch_size, 2 * num_beams)

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor") # beam上索引 (batch_size, 2 * num_beams)
            next_tokens = next_tokens % vocab_size  # vocab_size上的索引 (batch_size, 2 * num_beams)

            # stateless
            beam_outputs = beam_scorer.process(
                decoder_input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.config.pad_token_id,
                eos_token_id=self.config.eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"] # (batch_size * num_beams)
            beam_next_tokens = beam_outputs["next_beam_tokens"] # (batch_size * num_beams)
            beam_idx = beam_outputs["next_beam_indices"] # (batch_size * num_beams)

            # 更新最优路径 (batch_size*num_beams, cur_len)
            decoder_input_ids = torch.cat([decoder_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(outputs, decoder_input_ids, model_kwargs)

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            max_length=model_kwargs['max_new_tokens'],
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
        )

        sequences = self.rm_prompt_token_ids(sequence_outputs["sequences"], input_ids)
        return sequences


    def stream_generate(self):
        """stream输出预测的结果  单条数据"""

# TODO 转移至model下
class EncDecGenerationMixin(GenerationMixin):

    def init_decoder_input_ids(self, input_ids: torch.Tensor):
        """起始decoder_input_ids"""
        return torch.ones((input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device) * self.config.bos_token_id        # (bs, 1)

    def prepare_inputs_for_generation(
        self,
        step: int,
        input_ids: torch.Tensor = None,
        decoder_input_ids = None,
        encoder_outputs = None,
        attention_mask = None,
        **model_kwargs
    ) -> dict:
        """用kv_cache时，只需要取当前时刻的token_id即可 enc-dec"""
        decoder_input_ids = decoder_input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "start_pos": step,
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
        }

    def _update_model_kwargs_for_generation(
            self,
            outputs: Seq2SeqLMOutput,
            decoder_input_ids,
            model_kwargs
    ) -> dict:
        model_kwargs['decoder_input_ids'] = decoder_input_ids
        model_kwargs['encoder_outputs'] = outputs.encoder_last_hidden_state
        model_kwargs['attention_mask'] = outputs.encoder_attention_mask
        return model_kwargs


class DecGenerationMixin(GenerationMixin):

    def init_decoder_input_ids(self, input_ids: torch.Tensor):
        """起始decoder_input_ids
        """
        self.prompt_length = input_ids.size(1)
        return input_ids

    def update_token_type_ids(self, token_type_ids, *args):
        if token_type_ids is None:
            return None
        else:
            # 部分gpt模型有token_type_ids
            return token_type_ids[:, -1:]

    def prepare_inputs_for_generation(
        self,
        step: int,
        input_ids: torch.Tensor = None,
        decoder_input_ids = None,
        is_first_forward: bool = True,
        token_type_ids: torch.Tensor = None,
        **model_kwargs
    ) -> dict:
        """用kv_cache时，只需要取当前时刻的token_id即可"""
        if not is_first_forward:
            token_type_ids = self.update_token_type_ids(token_type_ids, decoder_input_ids[:, -step:])  # token_type_ids处理
            decoder_input_ids = decoder_input_ids[:, -1:]       # (batch_size, 1)
            step += self.prompt_length-1

        attention_mask:torch.Tensor = model_kwargs.get("attention_mask", None)
        position_ids = model_kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # 非batch模式以及position_ids为None
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
        return {
            "input_ids": decoder_input_ids,
            "start_pos": step,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "token_type_ids": token_type_ids,
        }

    def _update_model_kwargs_for_generation(
            self,
            outputs: CausalLMOutput,
            decoder_input_ids,
            model_kwargs
    ) -> dict:
        model_kwargs['decoder_input_ids'] = decoder_input_ids
        model_kwargs['is_first_forward'] = False

        attention_mask = outputs.attention_mask
        if attention_mask is not None:
            # 非batch模式下attention_mask为None
            model_kwargs['attention_mask'] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return model_kwargs

    def rm_prompt_token_ids(self, decoder_input_ids: torch.LongTensor, input_ids: torch.LongTensor):
        """decode_only模型去掉prompt部分"""
        decoder_input_ids = [ids[len(input_ids[i]):] for i, ids in enumerate(decoder_input_ids.tolist())]
        return decoder_input_ids


# TODO 解码器外部如何轻松干预
class UnilmGenerationMixin(DecGenerationMixin):

    def update_token_type_ids(self, token_type_ids: torch.LongTensor, decoder_input_ids: torch.LongTensor):
        """_summary_

        Args:
            token_type_ids (_type_): (bs, seq_len)
            decoder_input_ids (_type_): (bs, decoder_len)

        Returns:
            _type_: (bs, seq_len + decoder_len)
        """
        token_type_ids = torch.cat(
            [token_type_ids, torch.ones_like(decoder_input_ids, device=decoder_input_ids.device)], dim=1
        )
        return token_type_ids


class BeamSearchScorer:
    """
    [`BeamSearchScorer`] implementing standard beam search decoding.
    每一个时间步选取前n个候选序列，将这些候选序列添加到下一个时间步的束中
    TODO 参考 XML 实现   https://github.com/facebookresearch/XLM/blob/main/xlm/model/transformer.py#L529
    """

    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: Optional[float] = 1.0,
        do_early_stopping: Optional[Union[bool, str]] = False,
    ):
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping

        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor, # 当前时间步路径 (batch_size*num_beams, cur_len)
        next_scores: torch.FloatTensor, #        (batch_size, 2*num_beams)
        next_tokens: torch.LongTensor,  # 列索引  (batch_size, 2*num_beams)
        next_indices: torch.LongTensor, # 行索引  (batch_size, 2*num_beams)
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None
    ) -> Tuple[torch.Tensor]:
        cur_len = input_ids.shape[-1] + 1

        device = input_ids.device
        batch_size, num_beams = self.batch_size, self.num_beams
        next_beam_scores = torch.zeros((batch_size, num_beams), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, num_beams), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, num_beams), dtype=next_indices.dtype, device=device)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.num_beams + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() in eos_token_id):

                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    if beam_token_rank >= self.num_beams:
                        continue

                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item()
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.num_beams:
                    break

            #  判断当前序列是否结束
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return {
                "next_beam_scores": next_beam_scores.view(-1), 
                "next_beam_tokens": next_beam_tokens.view(-1), 
                "next_beam_indices": next_beam_indices.view(-1), 
        }


    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        max_length: int,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
    ) -> dict:
        batch_size = len(self._beam_hyps)

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # all open beam hypotheses are added to the beam hypothesis
            # beam hypothesis class automatically keeps the best beams
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # 选择最优路径
        sent_lengths = input_ids.new(batch_size)
        best = []
        best_scores = torch.zeros(batch_size, device=self.device, dtype=torch.float32)

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])

            best_hyp_tuple = sorted_hyps.pop()
            best_score = best_hyp_tuple[0]
            best_hyp = best_hyp_tuple[1]
            sent_lengths[i] = len(best_hyp)

            # append hyp to lists
            best.append(best_hyp)

            # append indices to list
            best_scores[i] = best_score

        # prepare for adding eos
        sent_lengths_max = sent_lengths.max().item() + 1
        sent_max_len = min(sent_lengths_max, max_length) if max_length is not None else sent_lengths_max
        decoded: torch.LongTensor = input_ids.new(batch_size, sent_max_len)

        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo

            if sent_lengths[i] < sent_max_len:
                # inserting only the first eos_token_id
                decoded[i, sent_lengths[i]] = eos_token_id[0]

        return {
                "sequences": decoded,
                "sequence_scores": best_scores,
        }
        

# ref: https://github.com/facebookresearch/XLM/blob/main/xlm/model/transformer.py#L705
class BeamHypotheses:
    def __init__(self, num_beams: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = [] # 保存解码出的路径
        self.worst_score = 1e9

    def __len__(self):
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float):
        """_summary_

        Args:
            hyp (torch.LongTensor): 待添加的路径
            sum_logprobs (float): 当前路径得分
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty) # 长度惩罚
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        判断最优路径集合是否结束
        如果最新得分比worst_score还低，则结束该序列
        """
        if len(self) < self.num_beams:
            return False

        elif self.early_stopping:
            return True
        else:
            return self.worst_score >= best_sum_logprobs / cur_len ** self.length_penalty
        

