# Measuring Zero-Shot MATH Performance

- 使用了 GSM8K 数据集

## Zero-shot MATH Baseline

- (b)
    - 统计：both 1: 0; format 1, answer 0: 254; both 0: 1065
    - format reward 0：大多数是模型的问题，有时完全没有标签，有时忘记 `</think>`标签，有时返回 '<think >'，有时少了一个空格；仅有少数几例模型回答 `</think>\n <answer>`  parser 误判为错误，parser似乎没有考虑到换行符等特殊符号的情况
    - format reward 1：只有两个是parser的错误，8个事模型的错误；模型问题在于只列式子不算数，或者忘记回答；2. parser的错误在于答案前后有特殊符号的时候判断不出来，比如 '$57500' 这个答案

- (c) Qwen 2.5 Math 1.5 B 在 GSM8K 上的 ero-shot baseline 表现极差，没有一个 answer 在 parser 的检查下为1，大多数格式不符合要求，答案大多数在胡说