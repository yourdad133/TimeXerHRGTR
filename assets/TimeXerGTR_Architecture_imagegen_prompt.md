# TimeXerGTR 架构图生成提示词

对应单周期模式（普通周期记忆），以 PEMS08 的 M 模式配置为例。
代码依据：models/TimeXerGTR.py、models/GTR.py、models/TimeXer.py、layers/Embed.py、scripts/baseline_gtr/TimeXerGTR_PEMS08_M.sh。

使用内置图像生成工具；工具未提供模型版本选择，无法核验用户所称的 image2.5。

实际提交的完整提示词：

Use case: scientific-educational / infographic-diagram.
Create ONE polished, accurate neural network architecture figure for the user's actual code, not an imagined method. Wide landscape approximately 16:9, high resolution, pure white background, flat vector-style scientific graphics, crisp Chinese sans-serif typography, generous spacing, straight orthogonal arrows. Palette: blue for observed inputs and TimeXer, teal for learnable periodic memory and GTR, amber for variable-context tokens, violet for learnable global tokens, green for prediction. No 3D, no gradients, no watermark, no decorative icons. Use Chinese labels plus the exact English module names in smaller type. Every arrow must end on its intended module. No overlapping text or tiny paragraphs.

Title verbatim: "TimeXerGTR 模型结构图"
Subtitle verbatim: "PEMS08 · 多变量预测（M）· 单周期记忆"
The figure has an upper panel A, and two lower detail panels B and C. Show the entire end-to-end graph in A, with B expanding GTR and C expanding one TimeXer encoder layer. The graph below is authoritative.

PANEL A title "A  整体结构".
Left input "历史窗口 X" with dimension "B × 96 × 170", entering "窗口归一化" with small "保存 μ, σ". After normalization produce "X_norm"; this node forks into TWO independent branches.
Main branch: X_norm → "GTR 增强" → "X_g = X_norm + γG" → "Patch 嵌入" → "TimeXer Encoder × 2" → "Flatten + Linear" → "反归一化" → "预测 Ŷ" dimension "B × H × 170".
Above GTR add a distinct learnable-memory table icon "可学习周期表 Q" dimension "288 × 170" → "周期对齐取窗" → q, connecting q into GTR. A small side input "周期索引 c" connects into "周期对齐取窗". Label the table explicitly as a trainable parameter, not stored past samples.
Beside Patch embedding show SIX little blue patch tokens plus ONE violet global token per variable, label "6 个 Patch + 1 个全局 Token". Patch label "长度 16 · 不重叠". A separate violet node "可学习全局 Token" connects to this token sequence. Token sequence enters encoder.
Second branch, from original X_norm before GTR: → amber "变量嵌入" ("DataEmbedding_inverted") → "变量上下文 Tokens", with arrow into the cross-attention input of TimeXer Encoder. Also show small "时间标记" entering variable embedding, and nearby note "PEMS08：零值标记". Do NOT feed X_g into this context branch.
Draw a dotted statistics bypass from normalization "μ, σ" to "反归一化"; bypass should run around outer edge, avoiding crossings. Re-normalization label "Ŷ = σ ⊙ Z + μ".
A compact configuration strip at bottom of A, verbatim: "T = 96   |   C = 170   |   L = 288   |   d_model = 512   |   Heads = 8   |   H ∈ {12, 24, 48, 96}".

PANEL B title "B  GTR：周期记忆与局部输入融合".
This is an expansion of the GTR augmentation part, not an extra downstream stage.
A small teal memory grid Q feeds a cycle-alignment box whose exact math reads "p_j = (c − T + j) mod L" and "q_j = Q[p_j]". Under formula small "c 为首个预测时刻；j = 0,…,T−1".
q → "时间线性映射" small "Linear(96 → 96)" → mapped q.
X_norm bypasses this linear map and feeds "Stack" as its other input. Stack has two visible layers, labeled "局部输入" and "周期查询".
Stack → "共享 Conv2D" small "kernel = 2 × 25" → "Dropout 0.1" → "G".
G → a multiplier node "× γ" → a circular plus node, whose other input is a separate skip arrow from X_norm → "X_g".
Small text "γ：可学习标量，初值 0.1". Shared Conv2D means one set of weights applied separately to every variable; do not depict channel mixing, cross-channel softmax, attention or nearest-neighbor search in GTR.

PANEL C title "C  单层 TimeXer Encoder".
Blue input token sequence with patch tokens and a violet final global token enters:
"时间 Self-Attention" → "Add & Norm" → then a split representation.
At split: only violet "全局 Token" goes into "Cross-Attention"; blue "Patch Tokens" bypass this cross-attention block.
Amber "变量上下文 Tokens" feeds the keys and values K,V of Cross-Attention; violet global token feeds Q. Label "Q：全局 Token" and "K,V：变量上下文".
Cross-Attention → "Add & Norm" → updated violet global token.
Recombine bypassed patch tokens and updated global token at "拼接 Tokens" → "FFN" small "512 → 2048 → 512" → "Add & Norm" → output token sequence.
Make clear only global tokens perform variable cross-attention and temporal self-attention operates across the patch-plus-global sequence within each variable. This layer is repeated twice in A. Use thin local skip arrows for Add & Norm only where legible.

Final small legend: blue "局部时序" / teal "周期记忆" / amber "变量上下文" / violet "全局 Token". Final scope note verbatim: "本图对应 普通周期记忆 的 TimeXerGTR".
Strict omissions: no autoregressive decoder, no nearest-neighbor historical retrieval, no GTR channel aggregation. All text must be legible and spelled correctly. Prioritize architecture fidelity and clear arrow topology over ornamental complexity.

