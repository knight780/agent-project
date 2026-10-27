# Web V1.0 修复与验收

## 验收口径

使用本地 sample.pdf（15 页、89 Chunks）、真实 MiniLM 与 Chroma 运行纯向量 Top-5。中文原问题未召回对应直接证据，英文版本命中，确认跨语言检索退化。

**修复后使用显式固定的 Mock 英文译文，回答也是 Mock，未调用真实 DeepSeek。** 此结果验证检索接入与引用校验，不代表真实云端翻译质量或回答准确率。保持原纯向量策略，未更换模型、Splitter 或索引算法。

## 逐题实际结果

### 1. Why did the authors choose sinusoidal positional encoding?

**修复前：原问题纯向量**

检索文本：Why did the authors choose sinusoidal positional encoding?

1. `chunk_35` · PDF 第 6 页 · 余弦距离 `0.480811`。摘要：learned and fixed [9]. In this work, we use sine and cosine functions of different frequencies: P E(pos,2i) = sin(pos/100002i/dmodel) P E(pos,2i+1) = cos(pos/100002i/dmodel) where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid
2. `chunk_58` · PDF 第 9 页 · 余弦距离 `0.606325`。摘要：suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace
3. `chunk_36` · PDF 第 6 页 · 余弦距离 `0.621906`。摘要：relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of P Epos. We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal ve
4. `chunk_34` · PDF 第 6 页 · 余弦距离 `0.680746`。摘要：Self-Attention (restricted) O(r · n · d) O(1) O(n/r) 3.5 Positional Encoding Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequen
5. `chunk_15` · PDF 第 2 页 · 余弦距离 `0.703130`。摘要：Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consu

Mock 流程结果：`answered`。作者选择正弦位置编码以支持相对位置学习，并可能外推到更长序列。[S1][S3]

- `[S1]` → sample.pdf → 第 6 页 → `chunk_35`
- `[S3]` → sample.pdf → 第 6 页 → `chunk_36`

**修复后：英文检索（固定 Mock 译文）**

检索文本：Why did the authors choose sinusoidal positional encoding?

1. `chunk_35` · PDF 第 6 页 · 余弦距离 `0.480811`。摘要：learned and fixed [9]. In this work, we use sine and cosine functions of different frequencies: P E(pos,2i) = sin(pos/100002i/dmodel) P E(pos,2i+1) = cos(pos/100002i/dmodel) where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid
2. `chunk_58` · PDF 第 9 页 · 余弦距离 `0.606325`。摘要：suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace
3. `chunk_36` · PDF 第 6 页 · 余弦距离 `0.621906`。摘要：relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of P Epos. We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal ve
4. `chunk_34` · PDF 第 6 页 · 余弦距离 `0.680746`。摘要：Self-Attention (restricted) O(r · n · d) O(1) O(n/r) 3.5 Positional Encoding Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequen
5. `chunk_15` · PDF 第 2 页 · 余弦距离 `0.703130`。摘要：Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consu

Mock 流程结果：`answered`。作者选择正弦位置编码以支持相对位置学习，并可能外推到更长序列。[S1][S3]

- `[S1]` → sample.pdf → 第 6 页 → `chunk_35`
- `[S3]` → sample.pdf → 第 6 页 → `chunk_36`

### 2. 论文为什么选择正弦位置编码？

**修复前：原问题纯向量**

检索文本：论文为什么选择正弦位置编码？

1. `chunk_55` · PDF 第 9 页 · 余弦距离 `0.785092`。摘要：Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to
2. `chunk_45` · PDF 第 7 页 · 余弦距离 `0.844164`。摘要：sentence pairs. Sentences were encoded using byte-pair encoding [ 3], which has a shared source- target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabula
3. `chunk_68` · PDF 第 10 页 · 余弦距离 `0.851875`。摘要：learning to align and translate. CoRR, abs/1409.0473, 2014. [3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V . Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017. [4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks f
4. `chunk_61` · PDF 第 9 页 · 余弦距离 `0.859516`。摘要：(section 5.4), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we 9
5. `chunk_52` · PDF 第 8 页 · 余弦距离 `0.876071`。摘要：On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used drop

Mock 流程结果：`model_refusal`。根据提供的论文证据，无法确认相关信息。请尝试询问论文中明确讨论的内容。


**修复后：英文检索（固定 Mock 译文）**

检索文本：Why did the authors choose sinusoidal positional encoding?

1. `chunk_35` · PDF 第 6 页 · 余弦距离 `0.480811`。摘要：learned and fixed [9]. In this work, we use sine and cosine functions of different frequencies: P E(pos,2i) = sin(pos/100002i/dmodel) P E(pos,2i+1) = cos(pos/100002i/dmodel) where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid
2. `chunk_58` · PDF 第 9 页 · 余弦距离 `0.606325`。摘要：suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace
3. `chunk_36` · PDF 第 6 页 · 余弦距离 `0.621906`。摘要：relative positions, since for any fixed offset k, P Epos+k can be represented as a linear function of P Epos. We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal ve
4. `chunk_34` · PDF 第 6 页 · 余弦距离 `0.680746`。摘要：Self-Attention (restricted) O(r · n · d) O(1) O(n/r) 3.5 Positional Encoding Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequen
5. `chunk_15` · PDF 第 2 页 · 余弦距离 `0.703130`。摘要：Here, the encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time. At each step the model is auto-regressive [10], consu

Mock 流程结果：`answered`。作者选择正弦位置编码以支持相对位置学习，并可能外推到更长序列。[S1][S3]

- `[S1]` → sample.pdf → 第 6 页 → `chunk_35`
- `[S3]` → sample.pdf → 第 6 页 → `chunk_36`

### 3. What results did Transformer achieve in machine translation?

**修复前：原问题纯向量**

检索文本：What results did Transformer achieve in machine translation?

1. `chunk_50` · PDF 第 8 页 · 余弦距离 `0.317151`。摘要：positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1. Label Smoothing During training, we employed label smoothing of value ϵls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score. 6 R
2. `chunk_52` · PDF 第 8 页 · 余弦距离 `0.405832`。摘要：On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used drop
3. `chunk_58` · PDF 第 9 页 · 余弦距离 `0.417316`。摘要：suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace
4. `chunk_3` · PDF 第 1 页 · 余弦距离 `0.450938`。摘要：training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. ∗Equal contribution.
5. `chunk_55` · PDF 第 9 页 · 余弦距离 `0.451818`。摘要：Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to

Mock 流程结果：`answered`。英法翻译取得 41.0 BLEU。[S2]

- `[S2]` → sample.pdf → 第 8 页 → `chunk_52`

**修复后：英文检索（固定 Mock 译文）**

检索文本：What results did Transformer achieve in machine translation?

1. `chunk_50` · PDF 第 8 页 · 余弦距离 `0.317151`。摘要：positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1. Label Smoothing During training, we employed label smoothing of value ϵls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score. 6 R
2. `chunk_52` · PDF 第 8 页 · 余弦距离 `0.405832`。摘要：On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used drop
3. `chunk_58` · PDF 第 9 页 · 余弦距离 `0.417316`。摘要：suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace
4. `chunk_3` · PDF 第 1 页 · 余弦距离 `0.450938`。摘要：training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. ∗Equal contribution.
5. `chunk_55` · PDF 第 9 页 · 余弦距离 `0.451818`。摘要：Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to

Mock 流程结果：`answered`。英法翻译取得 41.0 BLEU。[S2]

- `[S2]` → sample.pdf → 第 8 页 → `chunk_52`

### 4. Transformer 在机器翻译任务中取得了什么结果？

**修复前：原问题纯向量**

检索文本：Transformer 在机器翻译任务中取得了什么结果？

1. `chunk_4` · PDF 第 1 页 · 余弦距离 `0.633829`。摘要：the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the oth
2. `chunk_11` · PDF 第 2 页 · 余弦距离 `0.651398`。摘要：block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. Thi
3. `chunk_16` · PDF 第 3 页 · 余弦距离 `0.656329`。摘要：Figure 1: The Transformer - model architecture. The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively. 3.1 Encoder and Decoder Stacks Encoder
4. `chunk_55` · PDF 第 9 页 · 余弦距离 `0.662890`。摘要：Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to
5. `chunk_58` · PDF 第 9 页 · 余弦距离 `0.674078`。摘要：suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace

Mock 流程结果：`model_refusal`。根据提供的论文证据，无法确认相关信息。请尝试询问论文中明确讨论的内容。


**修复后：英文检索（固定 Mock 译文）**

检索文本：What results did Transformer achieve in machine translation?

1. `chunk_50` · PDF 第 8 页 · 余弦距离 `0.317151`。摘要：positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1. Label Smoothing During training, we employed label smoothing of value ϵls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score. 6 R
2. `chunk_52` · PDF 第 8 页 · 余弦距离 `0.405832`。摘要：On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used drop
3. `chunk_58` · PDF 第 9 页 · 余弦距离 `0.417316`。摘要：suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace
4. `chunk_3` · PDF 第 1 页 · 余弦距离 `0.450938`。摘要：training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data. ∗Equal contribution.
5. `chunk_55` · PDF 第 9 页 · 余弦距离 `0.451818`。摘要：Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to

Mock 流程结果：`answered`。英法翻译取得 41.0 BLEU。[S2]

- `[S2]` → sample.pdf → 第 8 页 → `chunk_52`

## 生成后误拒答复现

模拟返回：作者选择正弦位置编码以支持相对位置学习。[S1] 论文未提供实现代码。

原始全文正则判拒答：`True`；修复后状态：`answered`。

原规则在全文任意位置匹配“未提供”，将有引用的结论和后面的限制说明一起拒绝。新规则识别开头的明确拒答，保留无证据、无引用、越界编号及非规范编号拦截。真实 DeepSeek 是否在用户当时的请求中错误拒答，本次没有云端原始响应，无法确认。

## 引用显示

原始 sources 始终生成 [S1]、[S2] 等，未发现转换为“第二季/第四季”的后端代码。现将正文编号显示为行内代码；来源行使用转义后的 HTML，并添加 translate="no"、notranslate。只显示当前答案实际使用的来源，保留页码和 Chunk ID。

AppTest 验证编号、页码、Chunk ID、转义与禁止翻译标记。本次没有复现用户具体浏览器翻译扩展，浏览器翻译是待实机核查的解释，不能标记为已确认根因。

参考 [HTML translate 属性](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Global_attributes/translate)。

## 稳定性与完整回归

- PDF 大小、文件头和真实解析校验有效；不信任原文件名，内容哈希决定存储路径。
- 模型缓存和推理锁、Chroma 去重与索引复用保持；不同内容按哈希隔离目录。
- 清空聊天保留当前论文；换论文、移除或上传失败清理旧聊天、停用旧 Retriever。
- 独立会话的历史和显示文件名不混用；相同内容可以共用磁盘索引。
- 浏览器硬刷新会丢失会话，重新选同一 PDF 后复用磁盘索引。
- API、翻译、解析、检索失败均返回固定提示；空证据跳过生成；无内部路径、完整异常或密钥输出。
- 完整 pytest：**86 passed，1 条原有 langchain-community 弃用警告**。云端 SDK、翻译和问答全部 Mock；自动化测试无 API 费用。
- Streamlit AppTest 在允许本地 asyncio socketpair 的执行环境通过；编译检查、git diff --check 通过。

## 配置与复现

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
.\.venv\Scripts\python.exe -X utf8 diagnose_web_retrieval.py --compare --output data/web_v1_comparison.json
.\.venv\Scripts\python.exe -m pytest -q
```

中文翻译开关默认开启，可在侧栏关闭，或在 .env 设置 `RAG_TRANSLATE_CHINESE_QUERY=false` 改变默认值。中文开启时额外调用一次 DeepSeek；英文或关闭开关不翻译。翻译使用英文 system prompt，回答保留中文 system prompt。翻译失败不会静默回退到退化检索。

报告原始数据为 data/web_v1_comparison.json，含全部 Top-5 正文、距离、页码、原问题、检索译文、Mock 结果和引用映射；data/web_v1_baseline.json 保留修复前测得的检索结果。

## 剩余问题与建议

- 实际 DeepSeek 翻译、生成尚需手动测试四个问题；固定译文不能评估翻译泛化，其他译文会改变检索排序。
- 英文机器翻译问题虽命中英法 41.0 BLEU，但 Top-5 遗漏包含英德 28.4 BLEU 的 chunk_51；中文修复只是恢复到英文基线，并不保证答案完整。建议后续单独评估已有混合模式，本轮没有变更默认排名算法。
- 部分浏览器翻译扩展可能忽略 translate=no，需在用户浏览器开关翻译后核查 [S2]/[S4] 和来源行。
- 拒答判断仍是启发式规则，编号存在不能证明语义充分；未通过删除拒答保护提高表面回答率。

## 本轮修改文件

- streamlit_app.py：中文检索开关、引用渲染。
- app/web_service.py：查询与原问题分离、拒答误拦截修复、严格引用格式、结果状态。
- app/query_translation.py：可配置链路使用的查询翻译与结果校验。
- app/web_display.py：引用显示与 HTML 转义。
- app/rag/deepseek_client.py：仅增加可选 system_prompt 参数，默认中文行为保持。
- diagnose_web_retrieval.py：离线双语 Top-5 对照与 Mock 生成诊断。
- tests/test_web.py、tests/test_query_translation.py、tests/test_deepseek_client.py：回归测试。
- docs/STREAMLIT.md、本报告以及 data/web_v1_baseline.json、data/web_v1_comparison.json、data/web_v1_top5.txt：说明与诊断结果。

未改写 Loader、Splitter、Embedding 或 Chroma 核心实现；未增加 LangGraph/Multi-Agent，未 commit/push。
