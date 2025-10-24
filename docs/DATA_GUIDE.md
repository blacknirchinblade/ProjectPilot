# Using Real Datasets in ProjectPilot

## Where to Place Your Data
- Place your dataset files in the `data/` directory at the project root.
- Supported formats: CSV, JSON, Excel, images, etc. (depends on your generated project type).

## Updating the Generated Code
- After project generation, open the relevant data loading section in the generated code (see `output/<your_project>/`).
- Update the file path to point to your dataset in `data/`.
- Ensure your dataset matches the expected format (see code comments or docstrings).

## Example: Loading a Custom CSV
```python
import pandas as pd
df = pd.read_csv('data/your_dataset.csv')
```

## Tips for ML Projects
- For image or NLP datasets, update the data preprocessing section as needed.
- If using a public dataset, download it to `data/` and update the code path.
- For large datasets, consider using data generators or streaming.

## Advanced: Integrating New Data Sources
- Modify the data loading logic in the generated code to support databases, APIs, or cloud storage.
- Update requirements in `requirements.txt` if new libraries are needed.

---
*For more help, see the generated README or open an issue on GitHub.*
