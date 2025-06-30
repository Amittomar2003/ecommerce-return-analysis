 E-commerce Return Rate Reduction Analysis

## Objective
Analyze and reduce return rates in an e-commerce setting using machine learning and data visualization.

## Structure

- `data/` – Contains raw and processed datasets
- `scripts/` – Python code for data processing and modeling
- `outputs/` – Model predictions and results
- `dashboard/` – Power BI dashboard file (to be created)
- `notebooks/` – Optional: EDA and development notebooks

## Steps

1. Place raw files in `data/`
2. Run `scripts/model_train.py` to clean and train a logistic regression model
3. Output files will be generated in `outputs/`
4. Load the `return_predictions.csv` into Power BI for dashboard creation

## Output Files

- `return_predictions.csv`: Contains predicted return probabilities
- `high_risk_products.csv`: Products with >70% return risk

## Author
#
Amit Tomar
