### HW1: PM2.5 數值預測 (時序回歸)
- Implement linear regression using only numpy to predict the value of PM2.5
- 任務目標：透過過往 9 小時的環境觀測資料（如 PM2.5 原有數值），預測第 10 小時的 PM2.5 濃度
- 核心技術：
  - 資料預處理：處理非法字元（如 #, *, x）、使用 bfill() 與 ffill() 進行缺失值補全、特徵標準化（Normalization）
  - 模型實作：使用 NumPy 從底層實作 Linear Regression
  - 最佳化演算法：同時實作了 梯度下降法 (Gradient Descent) 與 閉式解 (Closed-form Solution/Normal Equation)
  - 正規化：加入 L2 正規化（Ridge Regression）以防止過擬合

### HW2: Amazon 評論情緒分析 (文字分類)
- Use DNN, LSTM, or BERT to perform a classification task.
- 任務目標：根據使用者的評論標題與內容，預測其給予的評分（1-5 分）
- 核心技術：
  - 預訓練模型：使用 BERT 作為 Encoder
  - 自然語言處理： 使用 BertTokenizer 處理文字、處理長度不一的序列（Padding/Truncation）、分析 Token 分佈
  - 訓練策略：採用 Ranger21 最佳化器（結合了 Lookahead 與 RAdam）、Early Stopping（早停機制）防止過擬合、DataParallel 多顯卡並行訓練
  - 評估指標：使用 Macro-F1 Score 來衡量多分類模型在不平衡資料下的表現

### HW3: 離群值偵測 (Outlier Detection)
- Use Random Forest and prediction uncertainty to perform outlier detection.
- 任務目標：透過模型對標籤預測的「不確定性」來找出資料集中的異常樣本
- 核心技術：
  - 整合學習：使用 Random Forest 作為基底分類器
  - 不確定性度量：利用 predict_proba 取得類別機率分佈，計算預測分佈與「均勻分佈」之間的 均方誤差 (MSE)
  - 原理：當模型對某樣本的預測機率趨近均勻分佈（即對每個類別都猶豫不決）時，其 MSE 較小，代表不確定性高，該樣本極可能是離群值（Outlier）

### Final Project: 新聞推薦系統 (點擊率預測)
- Leverage user interaction history to predict news click-through behavior for recommendation systems.
- 任務目標：根據使用者的歷史點擊紀錄與當前候選新聞，預測使用者點擊特定新聞的機率
- 核心技術：
  - 特徵編碼：使用 RoBERTa 提取新聞標題與摘要的語意向量（Embeddings），並預先序列化為 .pkl 檔以提升讀取效率
  - 模型架構：* LSTM 層： 處理使用者過往 35 則點擊新聞的時序特徵
    - User Representation：將 LSTM 的輸出經由全連接層轉化為使用者特徵向量
  - 交互機制： 計算 使用者向量與候選新聞向量的點積 (Dot Product)，以其相似度作為點擊機率
  - 工程實作： 自定義 Dataset 處理複雜的 history/impression 邏輯、使用 BCEWithLogitsLoss 處理二元分類任務
