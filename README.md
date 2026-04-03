# Supply Chain Analytics (SCA) Dashboard

This repository features a robust **Supply Chain Analytics** platform designed to transform raw procurement data into actionable insights. Built with Python and Streamlit, it integrates machine learning with linear programming to optimize supplier selection and monitor performance.

## 🚀 Key Features

* **Supplier Performance Ranking:** Uses a weighted scoring algorithm to rank vendors based on cost, quality, and reliability.
* **Procurement Optimizer:** Leverages mathematical optimization to allocate orders across suppliers while minimizing costs and meeting demand.
* **Predictive Analytics:** Implements machine learning models to forecast potential supply chain disruptions.
* **Interactive Visualizations:** High-fidelity charts showing lead-time distributions, cost breakdowns, and supplier geographical footprints.

## 📂 Repository Structure

* `app.py`: The main UI layer built with Streamlit.
* `supplier_optimizer.py`: The engine for procurement and resource allocation logic.
* `supplier_ranking.py`: Scripts for scoring and benchmarking supplier KPIs.
* `model.py`: Houses the machine learning logic for predictive modeling.
* `charts.py`: Custom Plotly/Matplotlib configurations for the dashboard.
* `supply_chain_clean_deploy_ready.csv`: The processed dataset powering the analytics.

## 🛠️ Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/atharvkanchan/sca.git
    cd sca
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the dashboard:**
    ```bash
    streamlit run app.py
    ```

---

## 🧩 How the Optimizer Works

The core intelligence of this project resides in the `supplier_optimizer.py` module. It operates on a multi-objective optimization framework:

### 1. The Objective
The primary goal is to minimize the **Total Cost of Ownership (TCO)**, defined as:
$$Total\ Cost = \sum (Unit\ Price \times Quantity) + Logistics\ Fees$$

### 2. Decision Constraints
The model ensures operational feasibility by respecting:
* **Capacity Limits:** No supplier is assigned more volume than their maximum verified output.
* **Service Levels:** Orders are prioritized toward suppliers with higher **On-Time Delivery (OTD)** rates.
* **Risk Mitigation:** Prevents over-reliance on a single source by distributing orders across the top $N$ ranked suppliers.

### 3. Ranking Integration
The optimizer dynamically pulls scores from `supplier_ranking.py`, which evaluates suppliers using a weighted matrix of:
* **Quality:** (1 - Defect Rate)
* **Timeliness:** (Days Late vs. Promised Lead Time)
* **Cost:** (Supplier Price vs. Category Benchmark)

---

## 🤝 Contributing
Contributions are what make the open-source community an amazing place to learn and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Project Link:** [https://github.com/atharvkanchan/sca](https://github.com/atharvkanchan/sca)
