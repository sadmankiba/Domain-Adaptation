# Results

## Dice Scores
Saved model = baseline 20 cls without_aug

| Test data                             | Global Dice    |  
|---------------------------------------|----------------|
| leftImg8bit/val/munster               |  0.70          |
| leftImg8bit_foggyDBF/val/munster 0.005|  0.68          |
| leftImg8bit_foggyDBF/val/munster 0.01 |  0.64          |
| leftImg8bit_foggyDBF/val/munster 0.02 |  0.55          |


Saved model = baseline 20 cls with aug

| Test data                             | Global Dice    |  
|---------------------------------------|----------------|
| leftImg8bit/val/munster               |  0.72          |
| leftImg8bit_foggyDBF/val/munster 0.005|  0.7           |
| leftImg8bit_foggyDBF/val/munster 0.01 |  0.67          |
| leftImg8bit_foggyDBF/val/munster 0.02 |  0.62          |
| leftImg8bit_foggyDBF/val/lindau 0.02  |  0.53          |
| leftImg8bit_foggyDBF/val/frankfurt 0.02|  0.62         |

Test time normalization (with aug)

| Test Files                    | BN Files | Foggy/Orig Mix | Shuffle | Num train iter | Global Dice |
| leftImg8bit_foggyDBF/val/ 0.02| fval     |       No mix   |  False  |   2            | 0.689       |
| leftImg8bit_foggyDBF/val/ 0.02| fval,ftrain|     No mix   |  False  |   4            | 0.689       |
| leftImg8bit_foggyDBF/val/ 0.02| fval,ftrain|     1        |  False  |   2            | 0.632       |
| leftImg8bit_foggyDBF/val/ 0.02| fval,oval  |     1        |  True   |   2            | 0.682       |
| leftImg8bit_foggyDBF/val/ 0.02| fval,oval  |     0.2      |  True   |   2            | 0.693       |
| leftImg8bit_foggyDBF/val/ 0.02| fval,oval  |     0.5      |  True   |   2            | 0.691      |


## Data

val/munster - 174
val/lindau  - 59
val/frankfurt - 267