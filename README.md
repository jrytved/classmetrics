# classmetrics
Multi-Class Classification Metrics with sklearn

![example output](https://github.com/jrytved/classmetrics/blob/main/metrics_screen.png?raw=true)

# Description
Not all of the classifications metrics included in sklearn.metrics are implemented for multi-class classification. This function computes a per-class accuracy, precision, recall, f1-score and specificity for multi-class classifiers.

If return_df, the function returns a dataframe with the metrics. The argument colnames takes the name of the classes. 

If return_df and style,  the function returns a styled dataframe with the string passed to title as the caption. All cells are colored by a red-green gradient in the range (0,1)

Otherwise the function returns accuracy, precision, recall, f1 and specificity with all being 1d numpy arrays of size n_classes. 

If used in a notebook the secondary function display_side_by_side, can be used to render several classification metric dataframes as HTML side-by-side.

# Code

## Computing Metrics

```python
from sklearn.metrics import multilabel_confusion_matrix, recall_score, f1_score, precision_score, accuracy_score
```

```python
def compute_classification_metrics(y_true: torch.tensor, y_pred: torch.tensor, return_df = False, colnames = None, style = False, title = False):
  """
  Computes accuracy, precision, recall, F1 score, specificity from multi-class classification predictions and labels.
  """

  def color_gradient(val):
    color = f'rgb({int(255 * (1 - val))}, {int(255 * val)}, 0)'
    return f'background-color: {color}'

  n_tasks = y_true.shape[1]

  cfm = multilabel_confusion_matrix(y_true, y_pred)

  # Precision, recall, F1-scores can be directly calculated with sklearn
  precision = precision_score(y_true, y_pred, average = None)
  recall = recall_score(y_true, y_pred, average = None)
  f1 = f1_score(y_true, y_pred, average = None)

  # Accuracy and specificity a are not implemented for multi-class tasks, so I'll calculate those from the CFM.

  accuracy, specificity  = np.zeros(n_tasks), np.zeros(n_tasks)

  for class_idx in np.arange(n_tasks):

    class_cfm = cfm[class_idx]
    tn, fp, fn, tp = class_cfm.ravel()
    
    _accuracy = (tp+tn) / (tp+tn+fp+fn)
    _specificity = tn / (fp+tn)

    accuracy[class_idx] = _accuracy
    specificity[class_idx] = _specificity

  if return_df:
    metric_names = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]
    stack = np.vstack([accuracy, precision, recall, f1, specificity])
    df = pd.DataFrame(stack, columns = colnames, index = metric_names).round(decimals=2)

    if not style:
      return df
    else:
      styled_df = df.style \
          .set_caption(title) \
          .applymap(color_gradient) \
          .format('{:.2f}') \
          .set_table_styles([{
          'selector': 'caption',
          'props': [
            ('color', 'black'),
            ('font-size', '24px'),
            ('font-weight', 'bold')
            ]
          }])
      return styled_df
  else:
    return accuracy, precision, recall, f1, specificity
```

## Displaying Metric DataFrames side-by-side.

```python
def display_side_by_side(dfs:list, tablespacing=5):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
    """
    output = ""
    for df in dfs:
      if type(df) == pd.io.formats.style.Styler:
        output += df.set_table_attributes("style='display:inline'")._repr_html_()
        output += tablespacing * "\xa0"
      else:
        output += df.style.set_table_attributes("style='display:inline'")._repr_html_()
        output += tablespacing * "\xa0"

    display(HTML(output))

```

# Examples

```python
dnn_train_metric_df = compute_classification_metrics(y_true=dnn_train_y, y_pred=dnn_train_preds_bool, return_df=True, colnames=target_names, style = True, title = "Tbl. 1: DNN Training Classification Metrics")
dnn_test_metric_df = compute_classification_metrics(y_true=dnn_test_y, y_pred=dnn_test_preds_bool, return_df=True, colnames=target_names, style = True, title = "Tbl. 2: DNN Test Classification Metrics")
dnn_val_metric_df = compute_classification_metrics(y_true=dnn_val_y, y_pred=dnn_val_preds_bool, return_df=True, colnames=target_names, style = True, title = "Tbl. 3: DNN Validation Classification Metrics")
```

```python
display_side_by_side([dnn_train_metric_df, dnn_test_metric_df, dnn_val_metric_df])
```
![example output](https://github.com/jrytved/classmetrics/blob/main/metrics_screen.png?raw=true)
