In the world of global enterprise, the Procure-to-Pay (P2P) cycle is a quiet battleground. Every day, massive corporations process millions of invoices. Human auditors can realistically only sample 1-2% of these transactions, and traditional, rule-based software tends to flag too many false positives, grinding the supply chain to a halt. 

The result? Billions of dollars are lost annually to "revenue leakage"—subtle 10-15% overcharges, duplicate billings, or missing Goods Receipt Notes (GRNs) that simply slip through the cracks.

It sounds like the perfect job for an LLM, right? **Wrong.** Here is the story of how we tried to build an AI financial auditor, why standard training methods failed spectacularly, and how we ultimately used Reinforcement Learning (GRPO) to build **DataHealth AI**, a virtual auditor with a mathematical intuition.

---

## 📌 The AI Hurdle: "Majority Class Collapse"

When we first tackled this problem, we assumed we could just fine-tune an LLM on historical accounting data. But we quickly ran into a wall known in machine learning as **Majority Class Collapse**.

In the real world, 90% of invoices are perfectly fine. When you train a Supervised Fine-Tuning (SFT) model on this data, the model quickly realizes that the mathematically safest way to minimize its loss function is to become lazy. It learns to simply rubber-stamp `APPROVE` on every single invoice. It ignores the nuanced math entirely.

We didn't need a model that memorized labels; we needed an active agent capable of hunting for discrepancies at scale.

---

## 🌍 Building the AP "Digital Twin"

To fix this, we stopped trying to *teach* the model and started forcing it to *experience* the AP environment. We built a custom simulation using Meta's PyTorch **OpenEnv** framework that acts as a digital twin of a corporate Accounts Payable department.

In this environment:
* Invoices are dynamically generated against a simulated `MARKET_DATABASE`.
* Vendors have persistent histories stored in a `VENDOR_LEDGER`.
* **The Core Tension:** The environment enforces a strict trade-off. Catching leaks saves the company money, but being overly aggressive and rejecting valid invoices destroys the **Vendor Trust Score**, which represents real-world supply chain velocity.

---

## 🧠 The RL Magic: GRPO & The "Strict Boss" Matrix

To cure the model's laziness, we transitioned to Reinforcement Learning—specifically **GRPO (Group Relative Policy Optimization)**. 

We used **Unsloth** for ultra-fast, memory-efficient LoRA training on top of the base `Llama-3.2-3B-Instruct` model. To prevent the model from collapsing into repetitive approvals, we injected high entropy during training (using a temperature of `1.3`), forcing it to explore mathematical boundaries.

### The Hidden 15% Rule
In our environment, a "Leak" is mathematically defined as any billed amount exceeding the true market benchmark by **>15%**, or any invoice lacking a valid GRN. **We never explicitly told the model this rule.** It had to discover this boundary purely through trial and error, guided by our "Strict Boss" reward matrix:

* **+2.0 (Massive Win):** Successfully using `FLAG_FOR_AUDIT` on a hidden 15%+ markup.
* **+0.5 (Standard):** Correctly approving a valid invoice.
* **-2.5 (Critical Penalty):** Approving an invoice with a missing GRN or high markup (Leaking Revenue).
* **-1.0 (False Alarm):** Flagging a perfectly fine vendor (Hurting the Vendor Trust Score).

---

## 🤖 Developing an Agentic "Theory of Mind"

The most fascinating outcome of this RL phase was watching the agent develop a rudimentary "Theory of Mind" regarding its environment. 

Through high-entropy exploration, the 3B model realized it was operating in a multi-agent ecosystem. It learned it couldn't just act as a calculator; it had to act as an investigator. The agent began autonomously using its `QUERY_HISTORY` tool to assess vendor intent before making financial decisions: *Is Vendor_104 making an honest mistake, or are they a historically bad actor probing our defenses?*

---

## 📈 Visualizing the Convergence: Training Evidence

By exposing the 3B model to this harsh environment, it successfully deduced the underlying supply-chain math. Here is the visual proof of that learning process:

### 1. The Smoothed Reward Trend (The "Aha!" Moment)
![smoothed_reward_curve](https://cdn-uploads.huggingface.co/production/uploads/69d3f66a037e731451d1523a/SBDvU4ARtvO0LMSmqAxEn.png)
> **Analysis:** Despite the high-temperature exploration, the 15-step moving average shows a definitive upward trajectory. Around Step 150, the agent stops randomly guessing and consistently starts hunting high-markup invoices to achieve the maximum `+3.0` payout per step.

### 2. The Raw Reward Distribution
![reward_curve](https://cdn-uploads.huggingface.co/production/uploads/69d3f66a037e731451d1523a/EZJeWqHv6mYhgh1H0iCSC.png)
> **Analysis:** The dense clustering of peaks hitting the upper bound on the right half of the graph visualizes the GRPO algorithm successfully shifting the policy toward the optimal auditing strategy.

### 3. Training Loss
![loss_curve](https://cdn-uploads.huggingface.co/production/uploads/69d3f66a037e731451d1523a/CXrCwZCF9vHkmEI1Afuml.png)
> **Analysis:** The surrogate policy objective remains highly stable (hovering near 0.0), indicating safe weight updates without catastrophic forgetting of its base instruction-following capabilities.

---

## 🚀 The Result: A Production-Ready Virtual Auditor

DataHealth AI successfully transformed from a lazy, text-predicting SFT model into a deterministic, strict financial auditor that interacts seamlessly with external APIs using JSON.

It balances the human elements of business (Vendor Trust) against raw mathematics, bringing true intelligent automation to the Procure-to-Pay cycle. 

**Want to see it in action or test the weights yourself?**
* 🔗 **Try the Live Environment API:** [AtulK29/LGDemo](https://huggingface.co/spaces/AtulK29/LGDemo)
* 🧠 **Download the Trained LoRA Weights:** [AtulK29/LeakGuard-3B-Auditor-L2](https://huggingface.co/AtulK29/LeakGuard-3B-Auditor-L2)
* 💻 **Explore the Code & Training Loop:** [GitHub Repository](https://github.com/Atul-Kumar29/LGworking)

