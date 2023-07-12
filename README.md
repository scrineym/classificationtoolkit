# ClassificationToolkit

A quick way of running multiple classifiers

## Installation 

To install run 

```shell
pip install git+https://github.com/scrineym/classificationtoolkit.git
```

## To run

ClassificationToolkit assumes you have a _preprocessed_ dataframe ready to be passed to a model


```python
from classificationtoolkit import ClassificationToolkit
import pandas as pd
import numpy as np


## Create a random dataframe for testing
df = pd.DataFrame(np.random.randint(0, 5, size=(100, 4)), columns=list('ABCD'))

# Setup CLTK
cc = ClassificationToolkit(df, 'D', './test')
cc.run_classifications()


```

## Reading the results

Inside the output dir you will see a classification report and saved model for each optimised for F1 score. 
