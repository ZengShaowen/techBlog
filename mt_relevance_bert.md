# 预训练技术在美团到店搜索广告的探索和实践


## 引言
美团到店搜索广告负责美团点评双平台站内搜索流量的商业变现，服务于到店餐饮、休娱亲子、丽人医美、酒店旅游等众多本地生活服务商家。我们的搜索广告业务上有这样一些特点，首先，在美团广告平台上广告主即自然商户供给，其次，在美团搜索场景中广告的展示样式非常原生，用户使用美团服务不会明显的区分广告和自然结果，搜索广告结果与用户意图的相关性尤为重要，否则用户体验损失导致搜索流量萎缩，进而导致平台商户流失的飞轮效应会更加明显。所以，我们的搜索广告平台除了优化流量变现效率等商业指标外，也需要重点优化用户体验，不断降低不相关广告对用户体验的损害，才能保证整个平台生态的长期健康发展。

在优化用户体验的目标下，如何正确的衡量用户体验，定义不相关广告是首要解决的问题。在搜索广告中，受结果列表页广告位置偏差、素材创意等因素影响，我们无法单一使用CTR等客观性指标来衡量用户体验，尤其首位、首屏等排序靠前广告的相关性问题被认为是影响用户体验的主要因素。因此，我们首先建立了美团场景下的搜索广告相关性标准和评估体系，主要通过例行采样和标注的方式对搜索关键词和广告结果进行相关、一般和不相关等分档标注，进而驱动我们的广告相关性模型和策略迭代，并采用广告排序前五位的Badcase率（即Badcase@5）作为搜索广告的相关性评估指标。

相关性模型主要负责搜索关键词和广告结果的相关性打分，类似于NLP中的文本匹配任务，我们最初采用ESIM[1]交互式模型，然而在实践过程中发现该模型对长文本表示的商户信息表征能力有限，正负样本区分能力不足，在过滤不相关广告的同时对相关广告的误伤率较高。而2018年底以来，以BERT[2]为代表的预训练语言模型在多项NLP任务上都取得了突破，所以我们也开始调研和探索预训练技术在搜索广告相关性上的应用。

针对搜索语义匹配任务，Google[3]和Bing[4]的搜索团队都基于BERT来编码Query和候选Doc，进而改善相关性效果。预训练模型在美团内部的NLP场景中也有不少落地实践，美团搜索已经验证了预训练模型在文本相关性任务上的有效性[5]。针对预训练在语义匹配任务中的应用，业界也提出不少方案。中科院计算所郭嘉丰等人提出PROP[6]和B-PROP[7]等针对搜索任务的预训练方法，主要思想是引入文档中代表词预测ROP(Representative wOrds Prediction)任务。纽约大学石溪分校曹庆庆等人提出DeFormer[8]分解预训练语言模型来做问答等语义匹配任务，在BERT的低层分别对问题和文档各自编码，再在高层部分拼接问题和文档的表征进行交互编码，让文档和问题在编码阶段尽可能地独立，从而提升模型的整体效率。百度刘璟等人提出RocketQA[9]和RocketQAv2[10]等面向端到端问答的检索模型训练方法，通过跨批次负采样、去噪的强负例采样以及数据增强技术大幅提升了双塔模型的效果。陈丹琦等人提出SimCSE[11]，采用自监督来提升模型的句子表示能力，从而提升语义匹配的效果。

另一方面，2020年至今，预训练从“大炼模型”迈向了“炼大模型”的阶段，通过设计先进的算法，整合尽可能多的数据，汇聚大量算力，集约化地训练超大模型，持续提升模型效果。不论是公开论文结果还是美团内部实践，均已证明更大规模的预训练模型能带来更好的下游任务效果。因此，我们与公司内美团NLP团队合作，尝试利用预训练模型来优化搜索关键词和广告结果的相关性，进一步降低首屏广告Badcase，提升用户体验。本文主要介绍了我们在训练样本上的数据增强，预训练（Pre-training）和微调（Fine-tuning）阶段的模型结构优化，应用到线上服务所进行的知识蒸馏等模型压缩工作，以及所取得的业务效果。

## 算法探索
在美团搜索广告场景中，相关性计算可以看做用户搜索意图(Query)和广告商户(POI/Doc)之间的匹配问题，具体实现中我们分别基于Query和POI的结构化信息匹配、文本匹配和语义匹配等方法进行打分并且进行分数融合。其中，结构化信息匹配主要是对Query分析结果与POI进行类目、属性等信息的匹配；文本匹配方面借鉴了搜索引擎中的传统匹配方法，包括Query和POI的term共现数、Query term覆盖率、TF-IDF、BM25打分等；语义匹配包括传统的隐语义匹配（如基于LDA或者Word2Vec计算相似度）和深度语义匹配方法，在广告相关性服务中我们采用学习能力更强的深度语义匹配模型。

深度语义匹配通常可以划分为表示型和交互型两类，表示型模型一般基于双塔结构分别得到两段输入文本的向量表示，最后计算两个文本表示的相似度，该方法的优点是Doc向量可提前离线计算缓存，匹配阶段计算速度快，适合线上应用，缺点是只在模型最后进行交互，对文本之间匹配关系学习不足；而交互型模型在初期对两段输入文本进行拼接，匹配阶段可以采用更复杂的网络结构，以学习文本间细粒度匹配关系，这种方法往往可以达到更高的精度，主要挑战在于线上应用的性能瓶颈。

美团搜索广告相关性服务的基线模型采用Transformer+ESIM的交互式模型结构，在满足性能的前提下有效缓解了部分相关性问题，但是实际应用中存在两点不足，1）商户除了基础门店信息外还关联了大量商品（团单），基线方法中直接将这些信息拼接成长文本作为POI输入，鉴于模型结构限制往往需要对POI文本进行截断，因而导致信息丢失；2) 基线模型对于长文本的表征能力有限，相关性判别能力不足，很难在控制变现效率影响的同时解决更多的Badcase。为了解决这些问题，我们基于BERT在数据和模型方面进行了一些探索和实践，下文展开介绍。

### 数据增强
鉴于BERT模型微调阶段所需数据量相比ESIM模型更少，并且对数据覆盖全面度、标签准确度、数据分布合理性等因素更为敏感，在进行模型结构探索前我们先按照如下思路产出一份可用性较高的数据。由于搜索广告涉及的业务众多且差异性大，包含的商品/服务种类更加多元，我们希望BERT的微调数据尽可能覆盖各个场景和主要服务，全部人工标注的人力和时间成本较高，而用户点击等行为能够一定程度反映出广告是否相关，所以训练数据主要基于曝光点击日志构造，对于部分困难样本加以规则及人工辅助。我们根据业务特性对训练数据的优化主要包括以下几点。

#### 正样本置信加权
正样本主要通过点击数据得到，我们对4个月内的Query-POI点击数据进行统计，并且基于曝光频次和点击率进行数据清洗以减少噪声。实际采样流程中，假设对于某个Query需要取N个POI构造N条正样本，采样过程中令POI被采样的概率与其点击数成正比，这样做主要是基于点击越多相关性越高的认知，既可以进一步保证标签置信，又有利于模型学习到POI之间不同的相关程度。在实验中我们也尝试了另外两个正样本采样方法：1）对某个Query随机取N个POI，2）对某个Query取点击最多的N个POI。实践发现方法1会采样到较多的弱相关样本，而方法2得到的大多为强相关样本，这两种方式均不利于模型拟合真实场景的数据分布。

#### 负采样分层
我们按照模型学习的困难程度从低到高设计了三种负样本采样方式：
- 全局随机负样本：大多为跨业务的负样本（比如烧烤和密室逃脱），模型学习最容易，可以有效过滤线上跨类目的恶劣Badcase；
- 一级类目内负样本：Query和POI属于相同一级类目（比如美食、丽人等），但是属于不同细化类目（比如祛痘和医学美容），这部分样本可以为模型学习增加一定难度，提高模型判别能力；
- 三级类目内负样本：Query和POI属于相同的细化类目，但是POI并不提供Query相关的服务（比如光子嫩肤和水光针商户），这部分属于困难负样本，可以提升模型对语义相近但服务不相关的Badcase的判别能力，更大程度保障用户体验；但是在三级类目下采样可能取到较多相关样本，所以这部分样本还需要经过基于服务核心词的规则过滤以及人工校验。

#### 采样平滑及分布一致性
- 采样平滑：在正样本构建过程中对Query采样频次做了平滑，避免高频Query采样过多导致模型忽略对中长尾Query样本的学习。
- 样本分布一致性：在负样本构建中，对于每种负样本均需要保证各Query出现概率与其在正样本中概率相等，避免样本分布不一致性导致模型学习有偏。

#### 文本关键词提取
美团搜索广告场景下，Query中可能包含地址词、品牌词、服务核心词等成分，Query文本一般较短，99%的Query长度小于10；POI的主要文本特征包括门店名称和商品信息，而广告主的商品数量普遍较多，直接拼接商品标题会导致POI文本过长，有26%的POI文本长度超过240。由于相关性模型的主要目标是学习Query和POI之间的服务相关性，大量冗余文本信息会影响模型性能和学习效果，我们对Query和POI文本进行如下处理以提取关键文本信息。
- 对于Query文本：基于命名实体识别和词权重结果过滤掉地址词、分店名等成分，保留服务核心词；
- 对于POI文本：对所有商品标题进行关键词抽取，得到一系列能反映商户核心服务的关键词，将其拼接作为POI文本，相比直接拼接原始商品文本，长度大幅下降，仅有5%的POI长度超过240，并且POI文本质量更高，模型学习效果更好。

最终我们的训练样本一共有50w条数据，涵盖餐饮、休娱、亲子和丽人等20个类目，其中正负样本比例为1:5，三种负样本比例为2:2:1。

### 模型优化
#### 基于多任务学习的多业务模型

由于美团搜索广告涉及餐饮、休娱亲子、丽人医美等大量业务场景，并且不同场景之间差异较大。从过去的实践经验可知，对于某个业务场景下的相关性优化，利用该业务数据训练的子模型相比利用全业务数据训练的通用模型往往效果更佳，但这种方法存在几个问题，1）多个子模型的维护和迭代成本更高，2）某些小场景由于训练数据稀疏难以正确学习到文本表示。受到多业务子模型优缺点的启发，我们尝试了区分业务场景的多任务学习，利用BERT作为共享层学习各个业务的通用特征表达，采用对应不同业务的多个分类器处理BERT输出的中间结果，实际应用中根据多个小场景的业务相似程度划分成N类，亦对应N个分类器，每个样本只经过其对应的分类器。多业务模型的主要优势在于能够利用所有数据进行全场景联合训练，同时一定程度上保留每个场景的特性，从而解决多业务场景下的相关性问题，模型结构如图1所示。
![@图1 多业务模型结构| center |450x0](https://github.com/ZengShaowen/techBlog/raw/master/figures/mt_relevance_bert/multi_bu_2.png)

#### 引入品类信息的预训练

由于美团搜索广告的样式较为原生，商品标题可能缺乏有效的结构化信息，有时仅根据Query和POI商品文本很难准确判断两者之间的语义相关性。例如【租车公司，<上水超跑俱乐部；宝马，奥迪>】，Query和POI文本的相关性不高，而该商户的三级品类是“养车-用车租车-租车”，我们认为引入品类信息有助于提高模型效果。为了更合理的引入品类信息，我们对BERT模型的输入编码部分进行改造，除了与原始BERT一致的Query、POI两个片段外，还引入了品类文本作为第三个片段，将品类文本作为额外片段的作用是防止品类信息对Query、POI产生交叉干扰，使模型对于POI文本和品类文本区别学习。图2为模型输入示意图，其中红色框内为品类片段的编码情况，Ec为品类片段的片段编码（Segment Embedding）。
![@图2 BERT输入部分引入POI品类信息 | center | ](https://github.com/ZengShaowen/techBlog/raw/master/figures/mt_relevance_bert/bert_input_category.png)

由于我们改变了BERT输入部分的结构，无法直接基于标准BERT进行相关性微调任务。我们对BERT重新进行预训练，并对预训练方式做了改进，将BERT预训练中用到的NSP（Next Sentence Prediction）任务替换为更适合搜索广告场景的点击预测任务，具体为“给定用户的搜索关键词、商户文本和商户品类信息，判断用户是否点击”。预训练数据采用自然及广告搜索曝光点击数据，大约6千万。

#### 模型优化的离线效果
为了清晰准确地反映模型迭代的离线效果，我们通过人工标注的方法构建了一份广告相关性任务Benchmark。基线ESIM模型、BERT模型以及本文提到的优化后BERT模型在Benchmark上的评估指标如表1所示。我们首先利用上文介绍的数据增强后的训练样本训练了MT-BERT-Base模型（12层768维），与ESIM模型相比，各项指标均显著提升，其中AUC提升6.6pp。在BERT模型优化方面，多任务学习和引入品类信息这两种方式均能进一步提升模型效果，其中引入品类信息的MT-BERT-Base模型效果更佳，相比标准的MT-BERT-Base模型AUC提升1.2pp。在BERT模型规模方面，实验发现随着其规模增长，模型效果持续提升，但是预训练和部署成本也相应增长，最终我们选取了大约3亿参数量的MT-BERT-Large模型（24层1024维），在同样引入品类信息的条件下，相比MT-BERT-Base模型AUC增长1.21pp。
|Model|Accuracy|AUC|F1-Score|
|:---:|:---:|:---:|:---:|
|ESIM（基线，旧训练数据）|67.73%|76.94%|72.62%|
|MT-BERT-Base|74.88%|82.65%|75.85%|
|MT-BERT-Base-多业务|75.41%|83.03%|76.49%|
|MT-BERT-Base-引入品类信息|77.33%|83.85%|77.93%|
|MT-BERT-Large-引入品类信息|77.87%|85.06%|79.14%|
<center> 表1 广告相关性任务模型优化迭代指标 </center>

## 应用实践
### 模型压缩
由于BERT模型的庞大参数量和前向预测耗时，直接部署上线会面临很大的性能挑战，通常需要将训练好的模型压缩为符合一定要求的小模型，业内常用模型压缩方案包括模型裁剪、低精度量化和知识蒸馏等。知识蒸馏[12]旨在有效地从大模型（教师模型）中迁移知识到小模型（学生模型）中，在业内得到了广泛的研究和应用，如HuggingFace提出的DistillBERT[13]和华为提出的TinyBERT[14] 等蒸馏方法，均在保证效果的前提下大幅提升了模型性能。经过在搜索等业务上的探索和迭代，美团NLP团队沉淀了一套基于两阶段知识蒸馏的模型压缩方案，包括通用型知识蒸馏和任务型知识蒸馏，具体过程如图3所示。在通用型知识蒸馏阶段，使用规模更大的预训练BERT模型作为教师模型，对学生模型在无监督预训练语料上进行通用知识蒸馏，得到通用轻量模型，该模型可用于初始化任务型知识蒸馏里的学生模型或直接对下游任务进行微调。在任务型知识蒸馏阶段，使用在有监督业务语料上微调的BERT模型作为教师模型，对学生模型在业务语料上进行领域知识蒸馏，得到最终的任务轻量模型，用于下游任务。实验证明，这两个阶段对于模型最终效果的提升都至关重要。

 ![@图3 两阶段知识蒸馏|center| 600*0](https://github.com/ZengShaowen/techBlog/raw/master/figures/mt_relevance_bert/two_stage_distill.png)

在美团搜索广告场景下，首先我们基于MT-BERT-Large（24层1024维）在大规模无监督广告语料上进行第一阶段通用型知识蒸馏，得到MT-BERT-Medium（6层384维）通用轻量模型，在下游的广告相关性任务上进行微调。MT-BERT-Medium属于单塔交互结构，如图4(a)所示。目前美团搜索广告系统中，每个Query请求会召回上百个POI候选，交互模型需要分别对上百个Query-POI对进行实时推理，复杂度较高，很难满足上线条件。常见解决方案是将交互模型改造成如图4(b)所示的双塔结构，即分别对Query和POI编码后计算相似度。由于待召回的大量POI编码可以离线完成，线上只需对Query短文本实时编码，使用双塔结构后模型效率大幅提升。我们使用通用型蒸馏得到的MT-BERT-Medium模型对双塔模型中Query和POI的编码网络进行初始化并且在双塔在微调阶段始终共享参数，因此本文将双塔模型记为Siamese-MT-BERT-Medium（每个塔为6层384维）。双塔结构虽然带来效率的提升，但由于Query和POI的编码完全独立，缺少上下文交互，模型效果会有很大损失，如表2所示，Siamese-MT-BERT-Medium双塔模型相比MT-BERT-Medium交互模型在相关性Benchmark上各项指标都明显下降。

![@图4 相关性模型结构对比|center|](https://github.com/ZengShaowen/techBlog/raw/master/figures/mt_relevance_bert/model_structure_comparison.png)


为了充分结合交互结构效果好和双塔结构效率高的优势，Facebook Poly-encoder[15]、斯坦福大学ColBERT[16]等工作在双塔结构的基础上引入不同复杂程度的后交互层（Late Interaction Layer）以提升模型效果，如图4(c)所示。后交互网络能提升双塔模型效果，但也引入了更多的计算量，在高QPS场景仍然很难满足上线要求。针对上述问题，在第二阶段任务型知识蒸馏过程中，我们提出了虚拟交互机制（Virtual InteRacTion mechanism，VIRT），如图4(d)所示，通过在双塔结构中引入虚拟交互信息，将交互模型中的知识迁移到双塔模型中，从而在保持双塔模型性能的同时提升模型效果。
![@图5 任务型知识蒸馏&虚拟交互|center|](https://github.com/ZengShaowen/techBlog/raw/master/figures/mt_relevance_bert/virtual_interact.png)

任务型知识蒸馏及虚拟交互的具体过程如图5所示。在任务型知识蒸馏阶段，我们首先基于MT-BERT-Large交互模型在业务语料上进行微调得到教师模型。由于学生模型Siamese-MT-BERT-Medium缺乏上下文交互，如图5(b)所示，注意力矩阵中的灰色部分代表了2块缺失的交互信息，我们通过虚拟交互机制对缺失部分进行模拟，计算公式如下为：
$$
\widetilde{\mathbf{M}}_{\mathbf{x}\rightarrow\mathbf{y}} = \operatorname{softmax}\left( \mathrm{Attention}( \widetilde{\mathbf{H}}_{\mathbf{x}}\widetilde{\mathbf{W}}_q,  \widetilde{\mathbf{H}}_{\mathbf{y}} \widetilde{\mathbf{V}}_k)\right) \\
\widetilde{\mathbf{M}}_{\mathbf{y}\rightarrow\mathbf{x}} = \operatorname{softmax}\left( \mathrm{Attention}( \widetilde{\mathbf{H}}_{\mathbf{y}}\widetilde{\mathbf{V}}_q,  \widetilde{\mathbf{H}}_{\mathbf{x}} \widetilde{\mathbf{W}}_k)\right) \\
$$
其中，$\widetilde{\mathbf{H}}_{\mathbf{x}}$和$\widetilde{\mathbf{H}}_{\mathbf{y}}$分别代表双塔模型中Query和POI表示，$\widetilde{\mathbf{W}}$和$\widetilde{\mathbf{V}}$分别是Query和POI进行编码时的模型参数，$\widetilde{\mathbf{M}}_{\mathbf{x}\rightarrow\mathbf{y}} $代表了$\widetilde{\mathbf{H}}_{\mathbf{x}}$到$\widetilde{\mathbf{H}}_{\mathbf{y}}$的注意力（即图5(b)右上角缺失部分），$\widetilde{\mathbf{M}}_{\mathbf{y}\rightarrow\mathbf{x}} $代表了$\widetilde{\mathbf{H}}_{\mathbf{y}}$到$\widetilde{\mathbf{H}}_{\mathbf{x}}$的注意力（即图5(b)左下角缺失部分）。而交互模型包含了Query和POI的全交互，计算公式为：
$$
\begin{align*}
\mathbf{S}  & =\mathrm {Attention}\left( \mathbf{H}_{}\mathbf{W}_q,\ \mathbf{H}_{} \mathbf{W}_k\right)  \\
& = \mathrm {Attention}\left( (\mathbf{H}_{\mathbf{x}},\mathbf{H}_{\mathbf{y}})_{}\mathbf{W}_q, \ (\mathbf{H}_{\mathbf{x}},\mathbf{H}_{\mathbf{y}})_{} \mathbf{W}_k\right) \\
& = \left[\begin{array}{l} \mathbf{S}_{\mathbf{x}\rightarrow\mathbf{x}} & \mathbf{S}_{\mathbf{x}\rightarrow\mathbf{y}}\\ \mathbf{S}_{\mathbf{y}\rightarrow\mathbf{x}} & \mathbf{S}_{\mathbf{y}\rightarrow\mathbf{y}} \end{array} \right]
\end{align*}
$$

其中，$\mathbf{H}$是交互模型中Query和POI的融合表示，可以分解为${\mathbf{H}}_{\mathbf{x}}$和${\mathbf{H}}_{\mathbf{y}}$，分别代表Query和POI，$\mathbf{W}$是模型参数。交互模型的自注意力矩阵可以分解为4个部分，其中$\mathbf{S}_{\mathbf{x}\rightarrow\mathbf{y}}$和$\mathbf{S}_{\mathbf{y}\rightarrow\mathbf{x}}$则是Query和POI之间的交互，也即双塔模型的缺失部分。我们对交互模型的交互矩阵和双塔模型的虚拟交互矩阵之间的L2距离进行最小化，从而将交互模型中的核心交互知识迁移到双塔模型中，计算过程为：
$$ \mathbf{M}_{\mathbf{x}\rightarrow\mathbf{y}} = \operatorname{softmax} \left( \mathbf{S}_{\mathbf{x}\rightarrow\mathbf{y}} \right) \\
  \mathbf{M}_{\mathbf{y}\rightarrow\mathbf{x}} = \operatorname{softmax} \left( \mathbf{S}_{\mathbf{y}\rightarrow\mathbf{x}} \right)$$
$$ \mathcal{L}_{\text {virt }} =\left\|\widetilde{\mathbf{M}}_{\mathbf{x}\rightarrow\mathbf{y}} - \mathbf{M}_{\mathbf{x}\rightarrow\mathbf{y}} \right\|_2 
    + \left\|\widetilde{\mathbf{M}}_{\mathbf{y}\rightarrow\mathbf{x}} - \mathbf{M}_{\mathbf{y}\rightarrow\mathbf{x}} \right\|_2 $$


我们对蒸馏阶段各个模型进行了Benchmark上的效果评估以及线上QPS=50时的性能测试，结果如表2所示。通过虚拟交互进行任务型知识蒸馏得到的任务轻量模型Siamese-MT-BERT-Medium相较于直接对通用轻量模型进行微调得到的同结构的Siamese-MT-BERT-Medium（w/o 任务型知识蒸馏）模型，各项效果指标明显提升，其中Accuracy提升1.18PP，AUC提升1.66PP，F1-Score提升1.54PP。最终我们对任务轻量模型Siamese-MT-BERT-Medium进行上线，相较于最初的MT-BERT-Large模型，线上推理速度提升56倍，完全满足线上服务的性能要求。
|Model	|模型规模 / 模型结构|	Accuracy|	AUC|	F1-Score|	参数量|	耗时|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|MT-BERT-Large	|24层1024维 / 交互|	77.87%|	85.06%|	79.14%|	340M	|227.5ms|
|**通用轻量模型**：MT-BERT-Medium 	|6层384维 / 交互|	77.62%|	84.79%|	78.63%|	21M	|16.8ms|
|Siamese-MT-BERT-Medium（w/o 任务型知识蒸馏）|	6层384维 / 双塔|	74.23%	|81.65%|	75.37%|	21M|	4.0ms|
|**任务轻量模型**： Siamese-MT-BERT-Medium	|6层384维 / 双塔|	75.41%|	83.31%|	76.91%|	21M|	4.0ms|
<center>表2 模型效果对比</center>



### 相关性服务链路优化
#### 相关性计算

为了更好地衡量召回结果的相关程度，除了基于模型得出的语义相关性之外，我们还计算了文本相关性、类目相关性等分数，并对所有分数进行融合得到最终的相关性分数。其中文本相关性的计算借鉴了搜索引擎场景常用的文本匹配方法，例如Query和POI的字符串包含关系、短语匹配数/匹配率、以及BM25分数等，另外文本匹配同时考虑了原串匹配、核心词匹配及同义词匹配等多维度指标；类目相关性主要基于Query的意图识别和商户类目信息进行匹配。分数融合模型可以选择LR或者GBDT等复杂度比较低的模型基于高质量标注数据集训练得到。

#### 相关性应用

通过模型结构和分数融合策略的迭代优化可以得到更加准确合理的相关性分数，但是在实际的相关性应用中，还需要紧密结合广告业务场景，综合考虑平台变现效率、用户体验、广告主供给及转化等多方面因素。基于“过滤恶劣Badcase”和“越相关的广告排序越靠前”两个基本思想，我们设计了几种相关性分数的具体应用方式：
- 过滤低质量广告：完全不相关的广告会严重影响用户体验，长期来看可能损害平台生态，需要进行过滤，另外考虑到不同召回策略和不同业务流量在变现效率及Badcase严重程度等方面的差异，过滤阈值被设计成召回策略*类目的二维可调节矩阵；
- 重排序参考相关性：在广告系统的竞价排序模块，在考虑点击率、转化率、交易额和出价等因素的同时，也需要考虑相关性分数；
- TOP位次相关性门槛：首位、首屏等排序靠前的广告结果对于用户体验至关重要，因此针对TOP位次设置了相关性门槛，进一步改善用户体验。

![@图6 相关性服务链路示意图|center| 450*0](https://github.com/ZengShaowen/techBlog/raw/master/figures/mt_relevance_bert/relevance_pipeline.png)

#### 模型部署
为了进一步提升服务性能并且能有效利用计算资源，模型部署阶段我们采用高频流量缓存、长尾流量实时计算的方案。对高频Query-POI对进行离线相关性计算并写入缓存，每日对新增或商品信息变化的Query-POI对进行增量计算并更新缓存，线上相关性服务优先取缓存数据，如果取不到则基于蒸馏后的任务轻量模型进行实时计算。对于输入相关性服务的Query-POI对，缓存数据的覆盖率达到90%以上，有效缓解了在线计算的性能压力。

![@图7 相关性分数离线/在线计算流程图|center| 550*0](https://github.com/ZengShaowen/techBlog/raw/master/figures/mt_relevance_bert/relevance_online.png)

线上实时计算的任务轻量模型使用TF-Serving进行部署，TF-Serving预测引擎支持使用美团机器学习平台的模型优化工具—ART框架（基于Faster-Transformer改进）进行加速，在将模型转为FP16精度后，最终加速比可达到5.5，数值平均误差仅为5e-4，在保证精度的同时极大地提高了模型预测效率。

### 线上效果
为了更加直接客观地反映线上广告相关性情况，我们建立了美团场景下的搜索广告相关性标准和评估体系，通过例行采样和标注的方式对搜索关键词和广告结果进行相关、一般和不相关等分档标注，采用排序前五位广告的Badcase率（即Badcase@5）作为搜索广告的相关性评估核心指标。除此之外，由于CTR能够通过用户行为间接反映广告的相关程度，并且便于在线上进行对比观测，而NDCG可以反映相关性分数用于广告列表排序的准确性，所以我们选取CTR和NDCG作为间接指标来辅助验证相关性模型迭代的有效性。我们对本文的优化进行了线上小流量实验，结果显示，实验组CTR提升1.0%，变现效率基本没有损失，并且经过人工评测，Badcase@5降低2.2pp，NDCG提升2.0pp，说明优化后的相关性模型能够对召回广告列表进行更加准确的校验，有效提升了广告相关性，从而给用户带来更好的体验。

下面列举了两个Badcase解决示例，图8(a)和8(b)分别包含了搜索“少儿古典舞”和“头皮spa”时的基线返回结果（左侧截屏）和实验组返回结果（右侧截屏），截图第一位为广告结果。在这两个示例中，实验组相关性模型将不相关结果“金益晨少儿艺术教育”和“莲琪科技美肤抗衰中心”过滤掉，相关广告得以曝光。

![@图8 Badcase解决示例|center|](https://github.com/ZengShaowen/techBlog/raw/master/figures/mt_relevance_bert/badcase.png)


## 总结与展望
本文介绍了预训练技术在美团到店搜索广告相关性上的实践，主要包括样本数据增强、模型结构优化、模型轻量化及线上部署等优化方案。在数据增强方面，为了基于曝光点击数据构造出适合美团广告场景下相关性任务的训练数据，我们构造了多种类型负样本，在采样时考虑正样本置信度、关键词频率平滑、正负样本均衡等因素，另外也对POI文本进行关键词抽取得到更加简短有效的文本特征。在模型结构优化方面，我们尝试了对不同业务场景做多任务学习，以及在BERT输入中引入品类文本片段这两种方案使模型更好地拟合美团搜索广告业务数据，同时利用规模更大的预训练模型进一步提升了模型的表达能力。在实践应用中，为了同时满足模型效果和线上性能要求，我们对中高频流量进行离线打分和缓存，并且利用MT-BERT-Large蒸馏得到的双塔模型进行线上预测以覆盖长尾流量。最终，在保证广告平台收入的前提下，有效降低了搜索广告Badcase率，提升了用户在平台的搜索体验。

目前广告相关性打分主要还是用于阈值门槛，目的是端到端的区分出不相关广告，从而快速降低广告Badcase。在此基础上，我们期望相关性打分能够继续提升区分相关和一般相关广告的能力，从而作为排序因子在重排中更好的平衡变现效率和用户体验指标，更准确的度量用户体验损失和变现效率提升的兑换关系。此外，在本地搜索类场景下，局部供给经常比较匮乏，实际召回效果对比全局供给的情况更依赖相关性打分的能力，所以我们依然需要在相关性模型上持续深入迭代，并且将支撑召回模型和策略的进一步优化。

在技术方向上，目前门槛阈值设置、广告长文本表达和业务知识融合等方面依然存在优化和提升空间：
1. **阈值搜索**：目前的阈值策略需要对每个类目分别调参，缺乏整体性且难以达到全局优化效果。未来我们计划将阈值搜索看作可变现流量上的最优化问题，在限定消耗损失及其他业务约束的条件下，找到一组门槛阈值使得整体Badcase解决最大化；

2. **特征表达**：目前POI特征主要采用商品标题的关键词抽取结果，但是POI文本仍然较长并且存在一些冗余信息，有必要对POI信息抽取方法继续探索，比如融合外部知识进行信息抽取，或者通过优化Transformer注意力机制使模型在判断相关性时更加关注某些重要短语或词项；

3. **联合优化**：Query和POI文本中的蕴含的类目信息、实体成分等对于判断相关性很有帮助，我们计划将相关性任务与搜索广告场景下其他任务联合优化，比如命名实体识别、Query类目识别等，期望通过引入辅助任务增强模型的学习能力，更全面准确地计算语义相关性。

##参考资料
[1] Chen, Qian, et al. "Enhanced lstm for natural language inference." arXiv preprint arXiv:1609.06038 (2016).

[2] Devlin, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." arXiv preprint arXiv: 1810.04805 (2018).

[3] Pandu Nayak, "Understanding searches better than ever before." Google blog (2019).

[4] Wenhao Lu, et al. "TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval." arXiv preprint arXiv: 2002.06275 (2020).

[5] 李勇, 佳昊, 杨扬等. BERT在美团搜索核心排序的探索和实践.

[6] Ma, Xinyu, et al. "PROP: Pre-training with Representative Words Prediction for Ad-hoc Retrieval." Proceedings of the 14th ACM International Conference on Web Search and Data Mining (2021).

[7] Ma, Xinyu, et al. "B-PROP: Bootstrapped Pre-training with Representative Words Prediction for Ad-hoc Retrieval." arXiv preprint arXiv: 2104.09791 (2021).

[8] Cao, Qingqing, et al. "DeFormer: Decomposing Pre-trained Transformers for Faster Question Answering." arXiv preprint arXiv:2005.00697 (2020).

[9] Qu, Yingqi, et al. "RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering." arXiv preprint arXiv: 2010.08191 (2021).

[10] Ren, Ruiyang, et al. "RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking." arXiv preprint arXiv: 2110.07367 (2021).

[11] Gao, Tianyu, et al. "SimCSE: Simple Contrastive Learning of Sentence Embeddings." arXiv preprint arXiv: 2104.08821 (2021).

[12] Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." arXiv preprint arXiv:1503.02531 (2015).

[13] Sanh, Victor, et al. "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter." arXiv preprint arXiv:1910.01108 (2019).

[14] Jiao, Xiaoqi, et al. "Tinybert: Distilling bert for natural language understanding." arXiv preprint arXiv:1909.10351 (2019).

[15] Humeau, Samuel, et al. "Poly-encoders: Transformer architectures and pre-training strategies for fast and accurate multi-sentence scoring." arXiv preprint arXiv:1905.01969 (2019).

[16] Khattab, Omar, and Matei Zaharia. "Colbert: Efficient and effective passage search via contextualized late interaction over bert." Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. (2020).

## 作者简介
曾邵雯、林春喜、钱晓俊、程佳、雷军，来自美团广告平台技术部。

杨扬、任磊、王金刚、武威，来自美团平台搜索与NLP部NLP中心。