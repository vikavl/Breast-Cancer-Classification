# Merge Jupyter notebooks
To merge multiple Jupyter notebooks (**.ipynb**) into one notebook without losing the outputs of individual cells, you can use the following approach. This approach ensures that the **cell outputs** (including figures, print statements, and results) are retained in the final merged notebook.

Here’s a detailed step-by-step guide:

### **Step 1: Install Required Packages**

To work with notebooks programmatically, you'll need the **`nbformat`** and **`nbmerge`** (optional) libraries. 

```bash
pip install nbformat nbmerge
```

- **`nbformat`**: Used to read, write, and manipulate Jupyter notebook files.
- **`nbmerge`**: A simple tool to merge Jupyter notebooks, which can help merge multiple `.ipynb` files into one, while keeping outputs.

---

### **Step 2: Manually Merge Notebooks**

If you want to manually merge your notebooks in a more controlled way, you can use **Jupyter Notebook's interface** and Python libraries like **`nbformat`** to read and write the notebook content, preserving the outputs.

#### **A. Merge Notebooks Using Jupyter Interface**:

1. **Open Jupyter Notebook**:
   - Start Jupyter in your terminal:
     ```bash
     jupyter notebook
     ```
   
2. **Open the Notebooks**:
   - Open each of the notebooks you want to merge.

3. **Copy & Paste Cells**:
   - You can **copy and paste** the cells from one notebook into the other. The **outputs** will be retained, as long as you copy both the input cells and their corresponding output cells.
   - **Note**: When copying cells, the outputs (e.g., figures or print outputs) will be pasted along with the code.

4. **Save the Merged Notebook**:
   - After copying and pasting the content of all the notebooks, save the final notebook.

#### **B. Merge Notebooks Using Python and `nbformat`**:

If you want to automate the merging process using Python, you can use **`nbformat`** to read each notebook and append the cells to a new notebook.

```python
import nbformat

def merge_notebooks(notebooks):
    # Create a new notebook object
    merged_notebook = nbformat.v4.new_notebook()

    # Loop through the list of notebook files
    for notebook_path in notebooks:
        # Load the individual notebook
        with open(notebook_path, 'r') as f:
            notebook = nbformat.read(f, as_version=4)

        # Append each cell from the current notebook to the new merged notebook
        merged_notebook.cells.extend(notebook.cells)

    # Write the merged notebook to a new file
    with open("merged_notebook.ipynb", "w") as f:
        nbformat.write(merged_notebook, f)

# List of your notebook files to merge
notebooks_to_merge = ["notebook1.ipynb", "notebook2.ipynb", "notebook3.ipynb"]

# Call the function to merge them
merge_notebooks(notebooks_to_merge)
```

#### **Steps for Merging with `nbformat`**:
1. **Read each notebook**:
   - Open each notebook using `nbformat.read()`.
2. **Append cells**:
   - Extract the cells from each notebook and append them to a new notebook object using `merge_notebook.cells.extend()`.
3. **Save the merged notebook**:
   - Use `nbformat.write()` to save the merged notebook to a file (`merged_notebook.ipynb`).

This method will preserve **cell outputs**, **metadata**, and **cell execution counts**, as long as the cells in the notebooks are intact.

---

### **Step 3: Using `nbmerge` (Optional Tool)**

If you prefer a simpler, command-line solution, you can use the **`nbmerge`** tool, which is designed to merge Jupyter notebooks while keeping outputs intact.

1. **Install `nbmerge`**:
   ```bash
   pip install nbmerge
   ```

2. **Merge Notebooks Using `nbmerge`**:
   After installing `nbmerge`, you can use it in the command line to merge notebooks:

   ```bash
   nbmerge notebook1.ipynb notebook2.ipynb notebook3.ipynb > merged_notebook.ipynb
   ```

   This will merge the notebooks `notebook1.ipynb`, `notebook2.ipynb`, and `notebook3.ipynb` into a new file `merged_notebook.ipynb`, keeping the outputs of each notebook.

---

### **Step 4: Execute the Merged Notebook (Optional)**

After merging the notebooks, you might want to **re-execute all cells** to ensure the outputs are fresh and up-to-date. You can do this using the **`nbconvert`** tool or programmatically through **`nbclient`**.

#### **A. Re-execute Using `nbconvert`**:
```bash
jupyter nbconvert --execute --inplace merged_notebook.ipynb
```

This will re-execute all the cells in `merged_notebook.ipynb` and update the outputs accordingly.

#### **B. Re-execute Programmatically with `nbclient`**:
```python
from nbclient import NotebookClient
import nbformat

# Load the merged notebook
with open("merged_notebook.ipynb", "r") as f:
    notebook = nbformat.read(f, as_version=4)

# Create a NotebookClient and execute the notebook
client = NotebookClient(notebook)
client.execute()

# Save the executed notebook
with open("executed_merged_notebook.ipynb", "w") as f:
    nbformat.write(notebook, f)
```

This code uses **`nbclient`** to programmatically execute the notebook, preserving the **outputs** and **execution order**.