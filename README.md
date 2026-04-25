---
title: LGDemo
emoji: 🚀
colorFrom: purple
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# 🛡️ LeakGuard: Intelligent RL Auditor

**Submission for MetaxScalarxPytorch Hackathon** An end-to-end supply-chain anomaly detection environment and a custom Reinforcement Learning (RL) agent, built on the Meta PyTorch **OpenEnv** framework.

🔗 **Quick Links:**
* **Trained Model Weights (LoRA):** [AtulK29/LeakGuard-3B-Auditor-L2](https://huggingface.co/AtulK29/LeakGuard-3B-Auditor-L2)
* **Hugging Face Blog/discussion:**[Curing "LLM Laziness": How We Built an RL Auditor to Catch Supply Chain Fraud](https://huggingface.co/spaces/AtulK29/LGDemo/discussions/1)
* **Live Environment API:** [AtulK29/LGDemo](https://atulk29-lgdemo.hf.space/docs) (Direct URL)
* **Hugging Face Space:** [Atulk29/LGDemo](https://huggingface.co/spaces/AtulK29/LGDemo)
* **Github Repository:** [Leakguarddemo](https://github.com/Atul-Kumar29/LGworking) 
* **Training Notebook:** [Kaggle RL Pipeline](https://www.kaggle.com/code/atulkumar29/notebook9ac1f1bb95) (> **Note to Judges:** If running this notebook manually in Colab/Kaggle, please restart the kernel manually after this installation cell to ensure dependencies load correctly. The programmatic restart command (`os._exit(00)`) has been commented out so it does not interrupt automated execution scripts.)

---

## 📌 The Core Problem: Revenue Leakage & "Majority Class Collapse"

**The Business Need:** In global supply chains, the Procure-to-Pay (P2P) cycle processes millions of invoices daily. Human auditors can only sample 1-2%, while rigid, rule-based software flags too many false positives. Consequently, billions of dollars are lost to subtle "revenue leakage"—hidden 10-15% vendor overcharges or missing Goods Receipt Notes (GRNs).

**The AI Hurdle:** When tasked with automating this, standard Supervised Fine-Tuning (SFT) models suffer from **"Majority Class Collapse."** Because 90% of real-world historical data is perfectly clean, the LLM simply gets lazy. It learns to rubber-stamp `APPROVE` on everything, ignoring the nuanced math to minimize its loss function. We needed an agent that actively hunts for discrepancies.

---

## 🌍 The Environment: Modeling the AP Ecosystem
To train the model, we built a custom PyTorch **OpenEnv** simulation that acts as a digital twin of a corporate Accounts Payable (AP) department. 
* It dynamically generates adversarial invoices against simulated market benchmarks.
* It tracks **Vendor Ledgers**, recording past reliability and flag history.
* **The Tension:** It enforces a strict trade-off. Catching leaks saves money, but being overly aggressive and rejecting valid invoices destroys the **Vendor Trust Score** (representing real-world supply chain velocity and friction).

---

## 🧠 Our Solution: GRPO, Entropy, & "Theory of Mind"
To cure the model's laziness, we abandoned SFT and utilized **Reinforcement Learning** via **GRPO (Group Relative Policy Optimization)**. 

### 1. Developing a "Theory of Mind"
Through high-entropy exploration, the agent develops a rudimentary "Theory of Mind" regarding its environment. It realizes it operates in a multi-agent ecosystem where vendors have distinct behaviors. It learns to use investigative tools (`QUERY_HISTORY`) to assess intent: *Is this vendor making an honest mistake, or are they a historically bad actor probing our defenses?*

### 2. The Model & Training Framework
* **Base Architecture:** `Llama-3.2-3B-Instruct`
* **Framework:** Unsloth for ultra-fast, memory-efficient LoRA training.
* **Temperature Injection:** We trained with high entropy (`temp=1.3`) to force the model to explore mathematical boundaries instead of collapsing into repetitive approvals.

### 3. The Core Logic & Reward Matrix
A "Leak" in our environment is mathematically defined as any billed amount exceeding the market benchmark by **>15%**, or lacking a valid GRN. We did *not* tell the model this rule; it had to discover it via a strict, non-clamped reward function:
* **+2.0 (Massive Win):** Successfully using `FLAG_FOR_AUDIT` on a hidden 15%+ markup.
* **+0.5 (Standard):** Correctly approving a valid invoice.
* **-2.5 (Critical Penalty):** Approving an invoice with a missing GRN or high markup (Leaking Revenue).
* **-1.0 (False Alarm):** Flagging a perfectly fine vendor (Hurting the Trust Score).

---

## 📈 Training Evidence & Convergence

By exposing the 3B model to this harsh environment, it successfully deduced the underlying supply-chain logic. 

### 1. The Smoothed Reward Trend (The "Aha!" Moment)

![Smoothed Reward Curve](smoothed_reward_curve.png)
> **Analysis:** Despite the high-temperature exploration, the 15-step moving average shows a definitive upward trajectory. Around Step 150, the agent stops randomly guessing and consistently starts hunting high-markup invoices to achieve the maximum `+3.0` payout per step.

### 2. The Raw Reward Distribution

![Raw Reward Curve](reward_curve.png)
> **Analysis:** The dense clustering of peaks hitting the upper bound on the right half of the graph visualizes the GRPO algorithm successfully shifting the policy toward the optimal auditing strategy.

### 3. Training Loss

![Training Loss Curve](loss_curve.png)
> **Analysis:** The surrogate policy objective remains highly stable (hovering near 0.0), indicating safe weight updates without catastrophic forgetting of its base instruction-following capabilities.

---

## ⚙️ Environment Action Space

The agent interacts with the `OpenEnv` FastAPI server using strict JSON outputs. 
1. **Standard Audit:** `{"invoice_id": <int>, "decision": "<APPROVE|FLAG_FOR_AUDIT|REJECT>"}`
2. **Negotiate:** `{"invoice_id": <int>, "decision": "NEGOTIATE", "discount_pct": <float>}`
3. **Investigate (Token Penalty):** `{"decision": "SEARCH_WEB", "item_name": "<string>"}`

---
## 🚀 The Result: A Production-Ready Virtual Auditor

LeakGuardAI successfully transformed from a lazy, text-predicting SFT model into a deterministic, strict financial auditor that interacts seamlessly with external APIs using JSON.

It balances the human elements of business (Vendor Trust) against raw mathematics, bringing true intelligent automation to the Procure-to-Pay cycle.

---

## 🚀 Evaluation Guide for Judges

We have optimized the environment for automated grading scripts. 

### 1. Automated Script Targeting (Important!)
Because Hugging Face wraps Spaces in an iframe, automated scripts hitting `huggingface.co/spaces/...` will receive a 404. **Your automated scripts must target the direct underlying container URL:**

👉 **Target API Base:** `https://atulk29-lgdemo.hf.space`

**Example Automated Step:**
```bash
curl -X POST "[https://atulk29-lgdemo.hf.space/step](https://atulk29-lgdemo.hf.space/step)" \
     -H "Content-Type: application/json" \
     -d "{\"invoice_id\": 1, \"decision\": \"APPROVE\"}"
```

### 2. Using our Provided Inference Script
You can test the actual trained 3B brain against the live environment by running our provided client:

```bash
# Install dependencies
pip install torch transformers peft requests

# Run the evaluation (Downloads the adapter from HF Hub)
python inference.py
```

### 3. Swagger UI Exploration
To manually inspect the API endpoints (`/reset`, `/step`, `/state`), visit the direct docs URL:
[https://atulk29-lgdemo.hf.space/docs](https://atulk29-lgdemo.hf.space/docs)
