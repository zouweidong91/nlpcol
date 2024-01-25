from typing import Optional, Union, List, Tuple

import torch
from torch import Size, Tensor


class GenerationMixin:
    """generation混入类"""

    def get_encoder_output(self, input_ids: torch.LongTensor) -> Tensor:
        """获取enc端的输出"""
        attention_mask = (input_ids != self.config.pad_token_id).long()
        enc_output = self.encoder(input_ids, attention_mask)
        return enc_output
        
    @torch.inference_mode()
    def generate(
        self, 
        input_ids: torch.LongTensor, # (batch_size, seq_len)
        mode: str = 'greedy_search', # 解码方式
        *args, 
        **kwargs
    ):
        self.eval() # eval模式
        """batch_generate"""
        assert mode in ('greedy_search', 'beam_search', 'do_sample')

        if mode == "beam_search":
            return self.generate_beam(input_ids, *args, **kwargs)

        batch_size = input_ids.shape[0]
        device = input_ids.device
        eos_token_id_tensor = torch.tensor([self.config.eos_token_id]).to(device)
        max_length = kwargs['max_length']

        enc_output = self.get_encoder_output(input_ids)
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * self.config.bos_token_id
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        for step in range(max_length):
            # 获取dec端的输出
            input_ids = self.prepare_inputs_for_generation(decoder_input_ids)

            outputs = self(
                decoder_input_ids=input_ids,
                encoder_outputs=enc_output.last_hidden_state,
                attention_mask=enc_output.attention_mask,
                start_pos = step
            )
            next_token_logits = outputs.lm_logits[:, -1, :] # 去最后一步的预测结果 (batch_size, vocab_size)
            next_tokens = getattr(self, mode)(next_token_logits, *args, **kwargs) # (batch_size)

            #  已经完成的序列，next_tokens强制设置为pad_token_id
            next_tokens = next_tokens * unfinished_sequences + self.config.pad_token_id * (1 - unfinished_sequences)

            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim=-1) # next_tokens.unsqueeze(1)

            # 当一个序列遇到eos_token_id时，设置为结束 
            # eos_token_id_tensor是为了兼容eos不唯一的情况
            # 这里已经完成的序列依然会继续进行计算，所以存在计算资源浪费的情况，kv_cache下，每次只计算一个token，所以影响并不大
            unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # 当一个batch内每条数据都遇到eos_token_id时，推理结束
            if unfinished_sequences.max() == 0:
                break

        return decoder_input_ids
        

    def prepare_inputs_for_generation(self, input_ids: Tensor) -> Tensor:
        """用kv_cache时，只需要取当前时刻的token_id即可"""
        return input_ids[:, -1:]


    def greedy_search(self, scores: Tensor, *args, **kwargs) -> Tensor:
        next_tokens = torch.argmax(scores, dim=-1) # (batch_size)
        return next_tokens


    def do_sample(
        self,
        scores: Tensor, # (batch_size, vocab_size)
        top_k: int = None, # 取概率最大的 K 个词
        top_p: float = None,  # 小于1。累积概率超过概率 p 的最小单词集中进行 一般取0.9左右
        temperature: float = 1, # 温度参数(0,2)，temperature<1时，拉大分值间的差异，softmax后分值大的采样概率更高
        *args, 
        **kwargs
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
        *args, 
        **kwargs
    ) -> Tensor:

        batch_size = input_ids.shape[0]
        device = input_ids.device
        batch_beam_size = batch_size * num_beams
        max_length = kwargs['max_length']

        # 对input_ids张量进行扩展
        input_ids = input_ids.repeat_interleave(num_beams, dim=0)  # (batch_size*num_beams, seq_len)  [0 0 1 1 2 2]
        enc_output = self.get_encoder_output(input_ids)
        decoder_input_ids = torch.ones((batch_beam_size, 1), dtype=torch.long, device=device) * self.config.bos_token_id  # (batch_size*num_beams, 1)

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

        for step in range(max_length):
            # 获取dec端的输出
            input_ids = self.prepare_inputs_for_generation(decoder_input_ids)

            outputs = self(
                decoder_input_ids=input_ids,
                encoder_outputs=enc_output.last_hidden_state,
                attention_mask=enc_output.attention_mask,
                start_pos = step
            )

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

            if beam_scorer.is_done:
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            max_length=self.config.max_seq_length,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
        )

        return sequence_outputs["sequences"]


    def stream_generate(self):
        """stream输出预测的结果  单条数据"""
        


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
    ) -> Tuple[torch.LongTensor]:
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
        

