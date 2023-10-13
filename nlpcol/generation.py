import torch
from torch import Size, Tensor

class GenerationMixin:
    """generation混入类"""

    @torch.inference_mode()
    def generate(self):
        pass
        
    
    def logits_processor(self, next_token_logits: Tensor) -> Tensor:
        """不同解码方式，此处不同"""
        return next_token_logits

    def prepare_inputs_for_generation(self, input_ids):
        """用kv_cache时，只需要取当前时刻的token_id即可"""

    def get_encoder_output(self, input_ids: torch.LongTensor) -> Tensor:
        """获取enc端的输出"""
        attention_mask = (input_ids != self.config.pad_token_id).long()
        enc_output = self.encoder(input_ids, attention_mask)
        return enc_output
        

    def greedy_search(
        self,
        input_ids: torch.LongTensor,

    ):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        eos_token_id_tensor = torch.tensor([self.config.eos_token_id]).to(device)

        enc_output = self.get_encoder_output(input_ids)
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=device) * self.config.bos_token_id
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        # tokens_scores

        for step in range(self.config.max_seq_length):
            outputs = self(
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=enc_output.last_hidden_state,
                attention_mask=enc_output.attention_mask
            )

            next_token_logits = outputs.lm_logits[:, -1, :] # 去最后一步的预测结果
            next_tokens_scores = self.logits_processor(next_token_logits)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            #  已经完成的序列，next_tokens强制设置为pad_token_id
            next_tokens = next_tokens * unfinished_sequences + self.config.pad_token_id * (1 - unfinished_sequences)

            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens[:, None]], dim=-1) # next_tokens.unsqueeze(1)

            # 当一个序列遇到eos_token_id时，设置为结束   TODO 换用llama的方式
            unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            # 当一个batch内每条数据都遇到eos_token_id时，推理结束
            if unfinished_sequences.max() == 0:
                break

        return decoder_input_ids




        



















































    def beam_search(self):
        pass

    def do_sample(self):
        pass
