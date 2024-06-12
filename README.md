# Deep-Generative-Model-Project
---

## 1 口语化框架

对于LLM在RE任务的现有zero-shot方案效果不好的问题，我们认为可能是因为LLM更加擅长QA类型的任务，而难以解决以结构化文本形式输入输出的RE任务，因此我们考虑将传统RE任务的形式转为QA任务形式进行实验。

### 口语化框架：Vanilla RE和QA4RE

![RE framework](/pic/REframework.png "RE framework")

在RE任务实验中，我们采用了两种零样本框架 [1]：Vanilla RE和QA4RE。在每种框架中，LLM将句子、实体1、实体2以及所有可能的关系作为输入，我们使用两种设置，一种有类型约束，另一种则没有。Vanilla RE简单地列出句子、实体和所有可能的关系，让LLM选择一种关系类型，而QA4RE则将关系转化为多选语言选项供LLM选择。与直接抽取关系的Vanilla RE框架不同，QA4RE将问题转换为生成式QA格式，理论上更加适合LLM，这种方法在英文数据上的有效性已被证明。



## 2 基于LoRA的微调

我们使用基于LoRA的参数高效微调方法在Llama2-7B 中文版上进行微调，以探索微调后LLM在关系抽取任务上的效果。

![LoRA](/pic/LoRA.png "LoRA")

其原理为对于LLM的预训练权重矩阵，使用一个低秩分解来表示参数更新

我们在中文数据集 DuIE 2.0上对中文版Llama2-7B模型基于LoRA进行微调，分别训练120和2500个steps，并对微调后的模型进行测试。



## 3 基于关系抽取的长文生成优化

在长文中，情节往往涉及多个角色和事件，它们之间存在着错综复杂的联系。针对长文生成任务的一致性问题，我们的思路是将关系抽取结合知识图谱构建应用于长文生成任务中，将每一段中的人物、事件和关系进行记录，从而提高文本的一致性和连贯性。这种方法允许我们在生成每个段落时，都能够考虑到前文所建立的上下文，确保人物之间的互动和事件的发展都符合逻辑，也确保新段落与已有内容的平滑衔接。

为此，我们设计基于关系抽取的长文生成优化框架，并在此基础上实现了长文生成原型系统，将关系抽取和知识图谱构建技术引入长文生成的传统流程中。我们设计的基于关系抽取的长文生成框架如下图所示：

![framework](/pic/framework.png "framework")                   

长文生成的过程中，我们的方案不再是一次性生成长文，而是将长文本分解成多个段落逐次生成。首先向LLM长文生成器中输入预期长文本的大纲和对应的提示词，生成出长文本中的一段文字。对于每一段生成的长文本中的每个句子   ，使用RE技术，对该句子中的关系实例三元组集合   中的每个实例抽取出实体关系三元组   。将每段文本新生成的实体关系三元组加入到一个知识图谱中。在下一段文本生成之前，将知识图谱随提示词一同输入LLM长文生成器，使得生成下一段落时仍具有前面的段落中精确的实体间关系。完整的长文文本生成就是循环上述过程，通过迭代的方式逐段生成文本。即： ![algorithm](/pic/algorithm.png "algorithm")

这种方式允许我们在生成每个段落之前，对前一段落中的人物关系、事件发展和关键信息进行回顾和分析，从而确保每个新段落都能够在前一段落的基础上继续发展，而不是重新开始，有助于维持故事的连贯性。这种方法还为文本的迭代改进提供了基础。在长文生成的过程中，我们可能会发现某些段落需要调整，以更好地符合整体故事的发展。通过记录每段中的关系，我们可以更容易地识别需要改进的地方，并进行有针对性的修改。