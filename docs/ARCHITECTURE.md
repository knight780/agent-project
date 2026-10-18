# AI Research Assistant Architecture

## 1. 离线建库

PDF -> Loader -> Splitter -> Embedding -> Chroma

## 2. 在线问答

Question -> Retriever -> Context -> LLM -> Answer

## 3. 当前开发阶段

T01：工程初始化

T02：PDF Loader

T03：Splitter + Embedding + Vector Store

T04：Retriever + Generator
