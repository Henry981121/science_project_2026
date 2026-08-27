# 融合層可以開跑了（v2_fusion）

## 先更新程式碼

分支 `exp`。關鍵的那個 commit 是 **`9c9b070`**（後面可能還有更新的，都一起拉下來就對了）。

```powershell
cd <專案根目錄>          # 有 .git 的那層，不是 handoff 資料夾

# 1. 看看自己在哪個分支、有沒有還沒存的改動
git status

# 2. 如果 git status 顯示有修改過的檔案（紅字的 modified），先收起來
#    沒有的話跳過這步
git stash push -m "pull 之前先收起來"

# 3. 抓下來
git fetch origin
git checkout exp          # 如果本來就在 exp 會顯示 "Already on 'exp'"
git pull origin exp

# 4. 確認拿到了 —— 這行要找得到東西
git log --oneline | Select-String 9c9b070

# 5. 如果第 2 步有 stash，現在放回來
git stash pop
```

第 4 步應該印出：

```
9c9b070 v2_fusion: 移除 GRL，改接 dct/clip/dinov2 handoff cache
```

沒印出任何東西就是沒拉到，別往下走。

<details>
<summary>如果 pull 卡住</summary>

**`Your local changes ... would be overwritten by merge`**
你本機改過同樣的檔案。先看是哪些：

```powershell
git status
```

如果是 `v2_fusion/` 裡的檔案、而且你沒有特意改過它們
（很可能只是跑過之後產生的差異），丟掉本機版本即可：

```powershell
git checkout -- v2_fusion/
git pull origin exp
```

**`CONFLICT (content): Merge conflict in ...`**
兩邊都改了同一個地方。想全部採用遠端的版本：

```powershell
git checkout --theirs v2_fusion/
git add v2_fusion/
git commit
```

（`--theirs` 只有在這個 conflict 狀態下才有效，平常用會報錯。）

**其他情況**：先把 `git status` 的輸出貼回來，不要硬解 —— 解錯會把東西弄丟。
</details>

---

## 改了什麼

1. **GRL 拿掉了。** 不做 domain adaptation 了，gradient reversal 那整條路徑刪除。
2. **接上你 8/26 的 handoff。** 三條流 dct / clip / dinov2，直接讀 `.npy`，不用轉檔。

模型 8.39M 參數（舊版 9.60M）。

---

## 跑法（三步，不要跳）

```powershell
cd <專案根目錄>
$CACHE = "C:\Users\harry\OneDrive\Desktop\science_project_v3\handoff_dct_clip_dinov2_20260826\feature_data\crop"

# 1. 靜態驗證。不用 cache、不用 GPU，30 秒
python -m v2_fusion.sanity

# 2. 只讀資料、不訓練。約 1 分鐘 —— 所有對齊問題都在這一步擋掉
python -m v2_fusion.train --cache-dir $CACHE --dry-run

# 3. 兩組對照
python -m v2_fusion.train --cache-dir $CACHE --preset all
```

需要 `torch pandas numpy scikit-learn`。特徵全載入記憶體約 **1.23 GB**。

---

## 每一步應該看到什麼

**第 1 步**：最後一行 `17/17 通過`。

**第 2 步**：

```
[train] n=109,193  fake=48,666 (44.6%)  real=60,527
  dims      : dct:192, clip:768, dinov2:1024
  invalid   : ? / 109,193 已剔除
[val] n=23,438 ...
[cross_generator_test] n=34,461 ...
  ⚠ source  : dcgan_unseen, fursona_gan, waifu_gan, wildfake_ddim, wildfake_other
              不在 source 類別空間，source 標籤為 -1
```

那個 ⚠ 是**正常的**，不是錯誤 —— 那 5 種只出現在 test，沒有 source 標籤。

> **n 對不上就停下來**，把輸出貼回來，先別跑第 3 步。
> （扣掉 valid mask 為 False 的列之後會略少於上面的數字，這是正常的。）

**第 3 步**：每組約數分鐘，結束會印一張對照總表，
結果在 `outputs/v2_fusion/{preset}/results.json`。

---

## 卡住的話

| 訊息 | 怎麼辦 |
|---|---|
| `找不到 .../dinov2/train.npy` | `--cache-dir` 指錯層了，要指到 `feature_data\crop` |
| `xxx.npy 的維度是 N，config.STREAM_DIMS 說是 M` | 維度跟 README 不一樣，貼回來，改一行就好 |
| `CSV 的 valid_all 與各流 .valid.npy 的交集不一致` | CSV 跟特徵可能不是同一次抽的，要查 |
| `出現不認得的 generator：xxx` | 有新的 generator 名稱，貼回來加進 config |
| CUDA OOM | 不太可能（模型很小），真的發生就加 `--batch-size 128` |

前四種在第 2 步就會擋下，不會浪費訓練時間。

---

## 另外三件想請你確認的（跟訓練並行，不用等）

### 1. train 和 cross_generator_test 有 704 筆檔名重疊

真圖側是 0（**最重要的那個已經乾淨了**），704 筆全在假圖側。
我猜是不同 generator 目錄下的檔名撞號（`dcgan/0001.png` vs `dcgan_unseen/0001.png`
是不同圖同檔名），但想確認一下：

```powershell
python -c "import pandas as pd, os, itertools; tr=pd.read_csv('feature_data/crop/index_train.csv'); te=pd.read_csv('feature_data/crop/index_cross_generator_test.csv'); tr['b']=tr.path.map(os.path.basename); te['b']=te.path.map(os.path.basename); ov=set(tr.b)&set(te.b); print('train side:', tr[tr.b.isin(ov)].generator.value_counts().to_dict()); print('test side:', te[te.b.isin(ov)].generator.value_counts().to_dict())"
```

兩側 generator 不同 → 撞號，沒事。

### 2. CLIP 的 768 維是哪裡來的？

CLIP ViT-L/14 的 hidden 是 1024，768 是它**自己訓練好的** visual projection 輸出。
如果是這個，完全沒問題。
但如果是我們自己加了一層 `Linear(1024→768)`，那層如果沒訓練過，
就等於把之前那個「隨機投影」的問題原封不動搬過來了。想確認是前者。

### 3. `wildfake` 同時在 train 和 test

train 有 `wildfake`、`dcgan`；cross_generator_test 有 `wildfake_ddim`、
`wildfake_other`、`dcgan_unseen`。7 個「未見」generator 裡有 3 個是訓練集
generator 的變體。

這**不一定是錯的**（WildFake 本身就是多生成器資料集，切不同子集很合理），
但寫成「未見生成器泛化」一定會被追問：是沒見過的**生成器**，
還是同一個資料集沒見過的**子集**？

想知道 `wildfake_ddim` / `wildfake_other` 用的生成器，跟 train 的 `wildfake`
是不是真的不同。有的話我們就有證據；沒有的話敘事要改成
「unseen generator families + held-out subsets」並分開報數字。

**這題不擋訓練**，跑出來的數字有效。只是在釐清之前，cross_generator_test
的數字先當內部參考，還不能寫成「對未見生成器的泛化能力」。

---

## 順帶

你把 locked `test` split 排除在 package 外面、要求模型凍結後才開，這個做法是對的，
而且是我們現在方法論上最強的一點。訓練跟調參都不要碰它。
