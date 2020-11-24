---
title: "IBM Fairness Pipeline"
date: 2020-06-14
tags: [machine learning, AI Fairness 360]
header:
  image: "/images/IBMFP.png= 250x"
excerpt: "Machine Learning, AI Fairness 360"
mathjax: "true"
---


### **Part One**

**AIF 360 Summary**

Machine learning models are used for high stakes decisions about people. It causes statistical discrimination, which is objectionable when it places certain privileged groups at systematic advantage and certain unprivileged groups at systematic disadvantage resulting in unwanted bias.

AI Fairness 360 (AIF360), an extensible open source toolkit for detecting, understanding, and mitigating algorithmic biases. The goals of AIF360 are to promote a deeper understanding of fairness metrics and mitigation techniques; to enable an open common platform for fairness researchers and industry practitioners to share and benchmark their algorithms; and to help facilitate the transition of fairness research algorithms to use in an industrial setting. 

AIF360 brings bias metrics, bias mitigation algorithms, bias metric explanations, and industrial usability together. By integrating these aspects, AIF360 can enable stronger collaboration between AI fairness researchers and practitioners, helping to translate the collective research results to practicing data scientists, data engineers, and developers deploying solutions in a variety of industries. 

Its pipeline consists of pre-processing, in-processing, and post-processing techniques to mitigate bias. Pre-processing techniques modify the original dataset to try and mitigate bias while in-processing techniques obtain the same goal via model training. Post-processing bias mitigation involves modifying the predictions of the model to promote fairness.

Binary Label Dataset Metrics shed insight on bias between unprivleged and privileged groups and offer many measures of bias. Classification Metrics go one step further and allow for the comparison of Binary Label Metrics.



### **Part Two**

**Detecting and mitigating age bias on credit decisions**
The goal of this tutorial is to introduce the basic functionality of AI Fairness 360 to an interested developer who may not have a background in bias detection and mitigation.

**Biases and Machine Learning**

A machine learning model makes predictions of an outcome for a particular instance. (Given an instance of a loan application, predict if the applicant will repay the loan.) The model makes these predictions based on a training dataset, where many other instances (other loan applications) and actual outcomes (whether they repaid) are provided. Thus, a machine learning algorithm will attempt to find patterns, or generalizations, in the training dataset to use when a prediction for a new instance is needed. (For example, one pattern it might discover is "if a person has salary > USD 40K and has outstanding debt < USD 5, they will repay the loan".) In many domains this technique, called supervised machine learning, has worked very well.

However, sometimes the patterns that are found may not be desirable or may even be illegal. For example, a loan repay model may determine that age plays a significant role in the prediction of repayment because the training dataset happened to have better repayment for one age group than for another. This raises two problems: 1) the training dataset may not be representative of the true population of people of all age groups, and 2) even if it is representative, it is illegal to base any decision on a applicant's age, regardless of whether this is a good prediction based on historical data.

AI Fairness 360 is designed to help address this problem with fairness metrics and bias mitigators. Fairness metrics can be used to check for bias in machine learning workflows. Bias mitigators can be used to overcome bias in the workflow to produce a more fair outcome.

The loan scenario describes an intuitive example of illegal bias. However, not all undesirable bias in machine learning is illegal it may also exist in more subtle ways. For example, a loan company may want a diverse portfolio of customers across all income levels, and thus, will deem it undesirable if they are making more loans to high income levels over low income levels. Although this is not illegal or unethical, it is undesirable for the company's strategy.

As these two examples illustrate, a bias detection and/or mitigation toolkit needs to be tailored to the particular bias of interest. More specifically, it needs to know the attribute or attributes, called protected attributes, that are of interest: race is one example of a protected attribute and age is a second.

**The Machine Learning Workflow**

To understand how bias can enter a machine learning model, we first review the basics of how a model is created in a supervised machine learning process.

![alt text](https://nbviewer.jupyter.org/github/IBM/AIF360/blob/master/examples/images/Complex_NoProc_V3.jpg)

First, the process starts with a training dataset, which contains a sequence of instances, where each instance has two components: the features and the correct prediction for those features. Next, a machine learning algorithm is trained on this training dataset to produce a machine learning model. This generated model can be used to make a prediction when given a new instance. A second dataset with features and correct predictions, called a test dataset, is used to assess the accuracy of the model. Since this test dataset is the same format as the training dataset, a set of instances of features and prediction pairs, often these two datasets derive from the same initial dataset. A random partitioning algorithm is used to split the initial dataset into training and test datasets.

Bias can enter the system in any of the three steps above. The training data set may be biased in that its outcomes may be biased towards particular kinds of instances. The algorithm that creates the model may be biased in that it may generate models that are weighted towards particular features in the input. The test data set may be biased in that it has expectations on correct answers that may be biased. These three points in the machine learning process represent points for testing and mitigating bias. In AI Fairness 360 codebase, we call these points pre-processing, in-processing, and post-processing.

**AI Fairness 360**

We are now ready to utilize AI Fairness 360 (aif360) to detect and mitigate bias. We will use the German credit dataset, splitting it into a training and test dataset. We will look for bias in the creation of a machine learning model to predict if an applicant should be given credit based on various features from a typical credit application. The protected attribute will be "Age", with "1" (older than or equal to 25) and "0" (younger than 25) being the values for the privileged and unprivileged groups, respectively. For this first tutorial, we will check for bias in the initial training data, mitigate the bias, and recheck. More sophisticated machine learning workflows are given in the author tutorials and demo notebooks in the codebase.

Here are the steps involved

**Step 1: Write import statements**
**Step 2: Set bias detection options, load dataset, and split between train and test**
**Step 3: Compute fairness metric on original training dataset**
**Step 4: Mitigate bias by transforming the original dataset**
**Step 5: Compute fairness metric on transformed training dataset**

**Step 1 Import Statements**

As with any python program, the first step will be to import the necessary packages. Below we import several components from the aif360 package. We import the GermanDataset, metrics to check for bias, and classes related to the algorithm we will use to mitigate bias.


```python
pip install aif360
```

    Requirement already satisfied: aif360 in /usr/local/lib/python3.6/dist-packages (0.3.0)
    Requirement already satisfied: scipy>=1.2.0 in /usr/local/lib/python3.6/dist-packages (from aif360) (1.4.1)
    Requirement already satisfied: numpy>=1.16 in /usr/local/lib/python3.6/dist-packages (from aif360) (1.18.5)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from aif360) (3.2.1)
    Requirement already satisfied: scikit-learn>=0.21 in /usr/local/lib/python3.6/dist-packages (from aif360) (0.22.2.post1)
    Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.6/dist-packages (from aif360) (1.0.4)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->aif360) (1.2.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->aif360) (2.8.1)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->aif360) (2.4.7)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib->aif360) (0.10.0)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn>=0.21->aif360) (0.15.1)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.24.0->aif360) (2018.9)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil>=2.1->matplotlib->aif360) (1.15.0)



```python
pip install -I tensorflow==1.13.1
```

    Collecting tensorflow==1.13.1
      Using cached https://files.pythonhosted.org/packages/77/63/a9fa76de8dffe7455304c4ed635be4aa9c0bacef6e0633d87d5f54530c5c/tensorflow-1.13.1-cp36-cp36m-manylinux1_x86_64.whl
    Collecting grpcio>=1.8.6
      Using cached https://files.pythonhosted.org/packages/cd/04/2b67f0a3645481235d5547891fd0e45e384f1ae5676788f24a7c8735b4e9/grpcio-1.29.0-cp36-cp36m-manylinux2010_x86_64.whl
    Collecting tensorflow-estimator<1.14.0rc0,>=1.13.0
      Using cached https://files.pythonhosted.org/packages/bb/48/13f49fc3fa0fdf916aa1419013bb8f2ad09674c275b4046d5ee669a46873/tensorflow_estimator-1.13.0-py2.py3-none-any.whl
    Collecting keras-preprocessing>=1.0.5
      Using cached https://files.pythonhosted.org/packages/79/4c/7c3275a01e12ef9368a892926ab932b33bb13d55794881e3573482b378a7/Keras_Preprocessing-1.1.2-py2.py3-none-any.whl
    Collecting gast>=0.2.0
      Using cached https://files.pythonhosted.org/packages/d6/84/759f5dd23fec8ba71952d97bcc7e2c9d7d63bdc582421f3cd4be845f0c98/gast-0.3.3-py2.py3-none-any.whl
    Collecting wheel>=0.26
      Using cached https://files.pythonhosted.org/packages/8c/23/848298cccf8e40f5bbb59009b32848a4c38f4e7f3364297ab3c3e2e2cd14/wheel-0.34.2-py2.py3-none-any.whl
    Collecting numpy>=1.13.3
      Using cached https://files.pythonhosted.org/packages/b3/a9/b1bc4c935ed063766bce7d3e8c7b20bd52e515ff1c732b02caacf7918e5a/numpy-1.18.5-cp36-cp36m-manylinux1_x86_64.whl
    Collecting six>=1.10.0
      Using cached https://files.pythonhosted.org/packages/ee/ff/48bde5c0f013094d729fe4b0316ba2a24774b3ff1c52d924a8a4cb04078a/six-1.15.0-py2.py3-none-any.whl
    Collecting protobuf>=3.6.1
      Using cached https://files.pythonhosted.org/packages/28/05/9867ef8eafd12265267bee138fa2c46ebf34a276ea4cbe184cba4c606e8b/protobuf-3.12.2-cp36-cp36m-manylinux1_x86_64.whl
    Processing /root/.cache/pip/wheels/7c/06/54/bc84598ba1daf8f970247f550b175aaaee85f68b4b0c5ab2c6/termcolor-1.1.0-cp36-none-any.whl
    Collecting tensorboard<1.14.0,>=1.13.0
      Using cached https://files.pythonhosted.org/packages/0f/39/bdd75b08a6fba41f098b6cb091b9e8c7a80e1b4d679a581a0ccd17b10373/tensorboard-1.13.1-py3-none-any.whl
    Processing /root/.cache/pip/wheels/8e/28/49/fad4e7f0b9a1227708cbbee4487ac8558a7334849cb81c813d/absl_py-0.9.0-cp36-none-any.whl
    Collecting keras-applications>=1.0.6
      Using cached https://files.pythonhosted.org/packages/71/e3/19762fdfc62877ae9102edf6342d71b28fbfd9dea3d2f96a882ce099b03f/Keras_Applications-1.0.8-py3-none-any.whl
    Collecting astor>=0.6.0
      Using cached https://files.pythonhosted.org/packages/c3/88/97eef84f48fa04fbd6750e62dcceafba6c63c81b7ac1420856c8dcc0a3f9/astor-0.8.1-py2.py3-none-any.whl
    Collecting mock>=2.0.0
      Using cached https://files.pythonhosted.org/packages/cd/74/d72daf8dff5b6566db857cfd088907bb0355f5dd2914c4b3ef065c790735/mock-4.0.2-py3-none-any.whl
    Collecting setuptools
      Using cached https://files.pythonhosted.org/packages/95/95/f657b6e17f00c3f35b5f68b10e46c3a43af353d8856bd57bfcfb1dbb3e92/setuptools-47.1.1-py3-none-any.whl
    Collecting werkzeug>=0.11.15
      Using cached https://files.pythonhosted.org/packages/cc/94/5f7079a0e00bd6863ef8f1da638721e9da21e5bacee597595b318f71d62e/Werkzeug-1.0.1-py2.py3-none-any.whl
    Collecting markdown>=2.6.8
      Using cached https://files.pythonhosted.org/packages/a4/63/eaec2bd025ab48c754b55e8819af0f6a69e2b1e187611dd40cbbe101ee7f/Markdown-3.2.2-py3-none-any.whl
    Collecting h5py
      Using cached https://files.pythonhosted.org/packages/60/06/cafdd44889200e5438b897388f3075b52a8ef01f28a17366d91de0fa2d05/h5py-2.10.0-cp36-cp36m-manylinux1_x86_64.whl
    Collecting importlib-metadata; python_version < "3.8"
      Using cached https://files.pythonhosted.org/packages/98/13/a1d703ec396ade42c1d33df0e1cb691a28b7c08b336a5683912c87e04cd7/importlib_metadata-1.6.1-py2.py3-none-any.whl
    Collecting zipp>=0.5
      Using cached https://files.pythonhosted.org/packages/b2/34/bfcb43cc0ba81f527bc4f40ef41ba2ff4080e047acb0586b56b3d017ace4/zipp-3.1.0-py3-none-any.whl
    [31mERROR: google-colab 1.0.0 has requirement six~=1.12.0, but you'll have six 1.15.0 which is incompatible.[0m
    [31mERROR: datascience 0.10.6 has requirement folium==0.2.1, but you'll have folium 0.8.3 which is incompatible.[0m
    [31mERROR: albumentations 0.1.12 has requirement imgaug<0.2.7,>=0.2.5, but you'll have imgaug 0.2.9 which is incompatible.[0m
    Installing collected packages: six, grpcio, numpy, absl-py, mock, tensorflow-estimator, keras-preprocessing, gast, wheel, setuptools, protobuf, termcolor, werkzeug, zipp, importlib-metadata, markdown, tensorboard, h5py, keras-applications, astor, tensorflow
    Successfully installed absl-py-0.9.0 astor-0.8.1 gast-0.3.3 grpcio-1.29.0 h5py-2.10.0 importlib-metadata-1.6.1 keras-applications-1.0.8 keras-preprocessing-1.1.2 markdown-3.2.2 mock-4.0.2 numpy-1.18.5 protobuf-3.12.2 setuptools-47.1.1 six-1.15.0 tensorboard-2.2.2 tensorflow-2.2.0 tensorflow-estimator-2.2.0 termcolor-1.1.0 werkzeug-1.0.1 wheel-0.34.2 zipp-3.1.0





```python
# Load all necessary packages
import sys
sys.path.insert(1, "../")  

import urllib
import numpy as np
np.random.seed(0)

from aif360.datasets import GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector
from aif360.algorithms.preprocessing import Reweighing, OptimPreproc
from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_german
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.algorithms.postprocessing.reject_option_classification import RejectOptionClassification
from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from sklearn.metrics import accuracy_score
import tensorflow as tf

from IPython.display import Markdown, display
```

    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint8 = np.dtype([("qint8", np.int8, 1)])
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint16 = np.dtype([("qint16", np.int16, 1)])
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      _np_qint32 = np.dtype([("qint32", np.int32, 1)])
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
      np_resource = np.dtype([("resource", np.ubyte, 1)])


**Step 2 Load dataset, specifying protected attribute, and split dataset into train and test**

In Step 2 we load the initial dataset, setting the protected attribute to be age. We then splits the original dataset into training and testing datasets. Although we will use only the training dataset in this tutorial, a normal workflow would also use a test dataset for assessing the efficacy (accuracy, fairness, etc.) during the development of a machine learning model. Finally, we set two variables (to be used in Step 3) for the privileged (1) and unprivileged (0) values for the age attribute. These are key inputs for detecting and mitigating bias, which will be Step 3 and Step 4.


```python
urllib.request.urlretrieve ("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data", "/usr/local/lib/python3.6/dist-packages/aif360/data/raw/german/german.data")

dataset_orig = GermanDataset(
    protected_attribute_names=['age'],           # this dataset also contains protected
                                                 # attribute for "sex" which we do not
                                                 # consider in this evaluation
    privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
    features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
)
dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
privileged_groups = [{'age': 1}]
unprivileged_groups = [{'age': 0}]

```

**Step 3 Compute fairness metric on original training dataset**

Now that we've identified the protected attribute 'age' and defined privileged and unprivileged values, we can use aif360 to detect bias in the dataset. One simple test is to compare the percentage of favorable results for the privileged and unprivileged groups, subtracting the former percentage from the latter. A negative value indicates less favorable outcomes for the unprivileged groups. This is implemented in the method called mean_difference on the BinaryLabelDatasetMetric class. The code below performs this check and displays the output, showing that the difference is -0.169905.


```python
metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())

metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Original testing dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())
```


#### Original training dataset


    Difference in mean outcomes between unprivileged and privileged groups = -0.169905



#### Original testing dataset


    Difference in mean outcomes between unprivileged and privileged groups = -0.006313


**Step 4 Mitigate bias by transforming the original dataset**

The previous step showed that the privileged group was getting 17% more positive outcomes in the training dataset. Since this is not desirable, we are going to try to mitigate this bias in the training dataset. As stated above, this is called pre-processing mitigation because it happens before the creation of the model.

AI Fairness 360 implements several pre-processing mitigation algorithms. We will choose the Reweighing algorithm [1], which is implemented in the Reweighing class in the aif360.algorithms.preprocessing package. This algorithm will transform the dataset to have more equity in positive outcomes on the protected attribute for the privileged and unprivileged groups.

We then call the fit and transform methods to perform the transformation, producing a newly transformed training dataset (dataset_transf_train).

[1] F. Kamiran and T. Calders, "Data Preprocessing Techniques for Classification without Discrimination," Knowledge and Information Systems, 2012.


```python
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf_train = RW.fit_transform(dataset_orig_train)
dataset_transf_test = RW.fit_transform(dataset_orig_test)
```

**Step 5 Compute fairness metric on transformed dataset**

Now that we have a transformed dataset, we can check how effective it was in removing bias by using the same metric we used for the original training dataset in Step 3. Once again, we use the function mean_difference in the BinaryLabelDatasetMetric class. We see the mitigation step was very effective, the difference in mean outcomes is now 0.0. So we went from a 17% advantage for the privileged group to equality in terms of mean outcome.


```python
metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
display(Markdown("#### Transformed training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
```


#### Transformed training dataset


    Difference in mean outcomes between unprivileged and privileged groups = 0.000000


**Summary**

The purpose of this tutorial is to give a new user to bias detection and mitigation a gentle introduction to some of the functionality of AI Fairness 360. A more complete use case would take the next step and see how the transformed dataset impacts the accuracy and fairness of a trained model. This is implemented in the demo notebook in the examples directory of toolkit, called demo_reweighing_preproc.ipynb. I highly encourage readers to view that notebook as it is generalization and extension of this simple tutorial.

There are many metrics one can use to detect the presence of bias. AI Fairness 360 provides many of them for your use. Since it is not clear which of these metrics to use, we also provide some guidance. Likewise, there are many different bias mitigation algorithms one can employ, many of which are in AI Fairness 360. Other tutorials will demonstrate the use of some of these metrics and mitigations algorithms.

As mentioned earlier, both fairness metrics and mitigation algorithms can be performed at various stages of the machine learning pipeline. We recommend checking for bias as often as possible, using as many metrics are relevant for the application domain. We also recommend incorporating bias detection in an automated continouus integration pipeline to ensure bias awareness as a software project evolves.

### **Part Three**

Above, we imported the German Dataset and split it into training and testing data. We also re-weighted the training and testing data to try and mitigate bias. On top of this pre-processing bias mitigation. mitigation, AIF360's pipeline includes in-processing and post-processing techniques. We use adversarial debiasing to train a model that improves fairness and reject option classification to modify the predictions from our model to even further improve fairness.

#### Adversarial Debiasing
This in-process technique limits the influence of the protected attribute in the decision by making it difficult to determine the protected attribute from the final prediction. This creates a model that reduces bias and promotes fairness for both groups.

##### Reject Option Classification
This post-processing technique looks at points near the decision boundary and switches predictions of unprivileged groups to favorable and privileged groups to unfavorable to further reduce bias from the model. It selects a region around the boundary with a pre-set margin and designed to catch points of highest uncertainty. This model doesn't just give equality for privileged and unprivileged groups, it goes a step further by giving preferential treatment to the unprivileged group.


```python
def print_binary_label_metrics(training_data, test_data):
    metric_train = BinaryLabelDatasetMetric(training_data, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    print("Training set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_train.mean_difference())

    metric_test = BinaryLabelDatasetMetric(test_data, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_test.mean_difference())



def print_classification_metrics(features, labels):
    classified_metric = ClassificationMetric(features, labels, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    print("Classification accuracy = %f" % classified_metric.accuracy())
    TPR = classified_metric.true_positive_rate()
    TNR = classified_metric.true_negative_rate()
    bal_acc_debiasing_test = 0.5*(TPR+TNR)
    print("Balanced classification accuracy = %f" % bal_acc_debiasing_test)
    print("Statistical parity difference = %f" % classified_metric.statistical_parity_difference())
    print("Disparate impact = %f" % classified_metric.disparate_impact())
    print("Equal opportunity difference = %f" % classified_metric.equal_opportunity_difference())
    print("Average odds difference = %f" % classified_metric.average_odds_difference())
    print("Theil_index = %f" % classified_metric.theil_index())

```

Above, we define two functions for printing performance metrics. We will use these functions many times throughout our pipeline.

###Binary Label Dataset Metric
Base class for all AIF datasets with binary outcomes. We use this method to compare the mean difference in outcomes between privileged and unprivileged groups, this sample statistics is a very telling glimpse into the amount of bias in favorable outcomes.

###Classification Metric
This class is used for comparing two binary label dataset metrics. In our case, we use this to compare our original datasets to the predictions made by our model. From this, we learn the following:

Classification accuracy - Total accuracy as a decimal point value.

Balanced classification accuracy - Mean between the rate of true positive and true negative predictions.
Statistical parity difference - The difference of the rate of favorable outcomes from the unprivileged group compared to the privileged group.

Disparate impact - The ratio of favorable outcomes for the unprivileged group compared to the privileged group.

Equal opportunity difference - The difference of true positive rates between the unprivileged and the privileged groups.


Average odds difference - Average difference of false positive rate and true positive rate between unprivileged and privileged groups.

Theil_index - Entropy of favorable outcomes for all individuals, privileged and unprivileged.




```python
display(Markdown("#### Mean differences on original dataset"))
print_binary_label_metrics(dataset_orig_train, dataset_orig_test)

display(Markdown("#### Mean differences on reweighted dataset"))
print_binary_label_metrics(dataset_transf_train, dataset_transf_test)

```


#### Mean differences on original dataset


    Training set: Difference in mean outcomes between unprivileged and privileged groups = -0.169905
    Test set: Difference in mean outcomes between unprivileged and privileged groups = -0.006313



#### Mean differences on reweighted dataset


    Training set: Difference in mean outcomes between unprivileged and privileged groups = 0.000000
    Test set: Difference in mean outcomes between unprivileged and privileged groups = -0.000000


As a baseline, we compare the mean differences in favorable outcomes in the privileged and unprivileged in both the weighted and unweighted groups. Looking at this, we can make two takeaways. 

First, there is a nearly 17% differerence in outcomes between original training data but only a 0.6% difference in ourcomes with the testing data. This is strange, our test size and shuffle must have put most of the unfavorable outcomes for unprivileged group in the training data but not the test data. This skewed data is likely to lead to unusual model behavior.

Second, as we noticed earlier using reweighting during pre-processing mitigates the bias in outcomes so there is no difference between the training and test data.

Next, we will use adversarial debiasing in our model. To compare performance, we will use two models: one with "debias" set to False and one with "debias" set to True. Ultimately, we should hope to notice a significant improvement in the second model.


```python
sess = tf.Session()
german_undebiased_model = AdversarialDebiasing(privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups, scope_name='plain_classifier', debias=False, sess=sess)
german_undebiased_model.fit(dataset_transf_train)
german_undebiased_train_pred = german_undebiased_model.predict(dataset_transf_train) 
german_undebiased_test_pred = german_undebiased_model.predict(dataset_transf_test)
display(Markdown("#### Mean differences on undebiased dataset"))
print_binary_label_metrics(german_undebiased_train_pred, german_undebiased_test_pred)


display(Markdown("#### German Dataset Undebiased training classification metrics"))
print_classification_metrics(dataset_transf_train, german_undebiased_train_pred)

display(Markdown("#### German Dataset Undebiased test classification metrics"))
print_classification_metrics(dataset_transf_test, german_undebiased_test_pred)

```

    
    WARNING: The TensorFlow contrib module will not be included in TensorFlow 2.0.
    For more information, please see:
      * https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md
      * https://github.com/tensorflow/addons
    If you depend on functionality not listed there, please file an issue.
    
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/aif360/algorithms/inprocessing/adversarial_debiasing.py:89: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
    Instructions for updating:
    Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
    epoch 0; iter: 0; batch classifier loss: 45.500259
    epoch 1; iter: 0; batch classifier loss: 56.475246
    epoch 2; iter: 0; batch classifier loss: 50.392849
    epoch 3; iter: 0; batch classifier loss: 52.382019
    epoch 4; iter: 0; batch classifier loss: 46.202183
    epoch 5; iter: 0; batch classifier loss: 37.198967
    epoch 6; iter: 0; batch classifier loss: 41.929314
    epoch 7; iter: 0; batch classifier loss: 24.895828
    epoch 8; iter: 0; batch classifier loss: 32.772141
    epoch 9; iter: 0; batch classifier loss: 31.921862
    epoch 10; iter: 0; batch classifier loss: 33.727707
    epoch 11; iter: 0; batch classifier loss: 28.410866
    epoch 12; iter: 0; batch classifier loss: 27.561531
    epoch 13; iter: 0; batch classifier loss: 23.532118
    epoch 14; iter: 0; batch classifier loss: 26.532749
    epoch 15; iter: 0; batch classifier loss: 28.492950
    epoch 16; iter: 0; batch classifier loss: 23.384710
    epoch 17; iter: 0; batch classifier loss: 24.818844
    epoch 18; iter: 0; batch classifier loss: 28.646439
    epoch 19; iter: 0; batch classifier loss: 27.074114
    epoch 20; iter: 0; batch classifier loss: 29.344862
    epoch 21; iter: 0; batch classifier loss: 25.727640
    epoch 22; iter: 0; batch classifier loss: 18.740362
    epoch 23; iter: 0; batch classifier loss: 13.052431
    epoch 24; iter: 0; batch classifier loss: 14.591496
    epoch 25; iter: 0; batch classifier loss: 14.442122
    epoch 26; iter: 0; batch classifier loss: 26.992331
    epoch 27; iter: 0; batch classifier loss: 21.503761
    epoch 28; iter: 0; batch classifier loss: 18.061150
    epoch 29; iter: 0; batch classifier loss: 14.524001
    epoch 30; iter: 0; batch classifier loss: 10.542807
    epoch 31; iter: 0; batch classifier loss: 14.506525
    epoch 32; iter: 0; batch classifier loss: 20.391949
    epoch 33; iter: 0; batch classifier loss: 16.909698
    epoch 34; iter: 0; batch classifier loss: 16.529778
    epoch 35; iter: 0; batch classifier loss: 15.015537
    epoch 36; iter: 0; batch classifier loss: 13.281448
    epoch 37; iter: 0; batch classifier loss: 15.790734
    epoch 38; iter: 0; batch classifier loss: 13.440853
    epoch 39; iter: 0; batch classifier loss: 16.196486
    epoch 40; iter: 0; batch classifier loss: 14.716669
    epoch 41; iter: 0; batch classifier loss: 8.707489
    epoch 42; iter: 0; batch classifier loss: 13.674579
    epoch 43; iter: 0; batch classifier loss: 14.616518
    epoch 44; iter: 0; batch classifier loss: 10.566471
    epoch 45; iter: 0; batch classifier loss: 13.739122
    epoch 46; iter: 0; batch classifier loss: 8.712069
    epoch 47; iter: 0; batch classifier loss: 8.475683
    epoch 48; iter: 0; batch classifier loss: 11.403507
    epoch 49; iter: 0; batch classifier loss: 9.488426



#### Mean differences on undebiased dataset


    Training set: Difference in mean outcomes between unprivileged and privileged groups = 0.000000
    Test set: Difference in mean outcomes between unprivileged and privileged groups = 0.000000



#### German Dataset Undebiased training classification metrics


    Classification accuracy = 0.700000
    Balanced classification accuracy = 0.500000
    Statistical parity difference = 0.000000
    Disparate impact = 1.000000
    Equal opportunity difference = 0.000000
    Average odds difference = 0.000000
    Theil_index = 0.057550



#### German Dataset Undebiased test classification metrics


    Classification accuracy = 0.700000
    Balanced classification accuracy = 0.500000
    Statistical parity difference = 0.000000
    Disparate impact = 1.000000
    Equal opportunity difference = 0.000000
    Average odds difference = 0.000000
    Theil_index = 0.057550


These results are very strange, we had the "debias" option set to False and yet all of our bias metrics indicate near-perfect fairness. Not only this but we have identical metrics in both our training and test datasets. Let's move on to our debiased model and see what the results look like.


```python
sess.close()
tf.reset_default_graph()
sess = tf.Session()

german_debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups, scope_name='debiased_classifier', debias=True, sess=sess)
german_debiased_model.fit(dataset_transf_train)
german_debiased_train_pred = german_debiased_model.predict(dataset_transf_train) 
german_debiased_test_pred = german_debiased_model.predict(dataset_transf_test)
display(Markdown("#### Mean differences on debiased German Dataset"))
print_binary_label_metrics(german_debiased_train_pred, german_debiased_test_pred)


display(Markdown("#### German Dataset Debiasing training classification metrics"))
print_classification_metrics(dataset_transf_train, german_debiased_train_pred)

display(Markdown("#### German Dataset Debiasing test classification metrics"))
print_classification_metrics(dataset_transf_test, german_debiased_test_pred)

```

    epoch 0; iter: 0; batch classifier loss: 110.634178; batch adversarial loss: 0.495368
    epoch 1; iter: 0; batch classifier loss: 77.749153; batch adversarial loss: 0.476633
    epoch 2; iter: 0; batch classifier loss: 46.263870; batch adversarial loss: 0.520275
    epoch 3; iter: 0; batch classifier loss: 56.376030; batch adversarial loss: 0.588132
    epoch 4; iter: 0; batch classifier loss: 31.395973; batch adversarial loss: 0.571279
    epoch 5; iter: 0; batch classifier loss: 55.675900; batch adversarial loss: 0.583158
    epoch 6; iter: 0; batch classifier loss: 52.526287; batch adversarial loss: 0.555581
    epoch 7; iter: 0; batch classifier loss: 37.205658; batch adversarial loss: 0.544708
    epoch 8; iter: 0; batch classifier loss: 40.441658; batch adversarial loss: 0.561097
    epoch 9; iter: 0; batch classifier loss: 52.100716; batch adversarial loss: 0.560176
    epoch 10; iter: 0; batch classifier loss: 56.181961; batch adversarial loss: 0.564439
    epoch 11; iter: 0; batch classifier loss: 52.194016; batch adversarial loss: 0.552071
    epoch 12; iter: 0; batch classifier loss: 45.740112; batch adversarial loss: 0.494347
    epoch 13; iter: 0; batch classifier loss: 44.464828; batch adversarial loss: 0.552037
    epoch 14; iter: 0; batch classifier loss: 38.324417; batch adversarial loss: 0.548966
    epoch 15; iter: 0; batch classifier loss: 39.677727; batch adversarial loss: 0.570425
    epoch 16; iter: 0; batch classifier loss: 28.671898; batch adversarial loss: 0.569311
    epoch 17; iter: 0; batch classifier loss: 38.778408; batch adversarial loss: 0.518938
    epoch 18; iter: 0; batch classifier loss: 39.149601; batch adversarial loss: 0.563781
    epoch 19; iter: 0; batch classifier loss: 50.730667; batch adversarial loss: 0.526900
    epoch 20; iter: 0; batch classifier loss: 42.124214; batch adversarial loss: 0.554088
    epoch 21; iter: 0; batch classifier loss: 38.671158; batch adversarial loss: 0.526211
    epoch 22; iter: 0; batch classifier loss: 40.361019; batch adversarial loss: 0.527656
    epoch 23; iter: 0; batch classifier loss: 27.985542; batch adversarial loss: 0.531238
    epoch 24; iter: 0; batch classifier loss: 42.027481; batch adversarial loss: 0.502777
    epoch 25; iter: 0; batch classifier loss: 32.113426; batch adversarial loss: 0.514754
    epoch 26; iter: 0; batch classifier loss: 30.024876; batch adversarial loss: 0.533605
    epoch 27; iter: 0; batch classifier loss: 32.668793; batch adversarial loss: 0.578583
    epoch 28; iter: 0; batch classifier loss: 41.538986; batch adversarial loss: 0.540321
    epoch 29; iter: 0; batch classifier loss: 28.629177; batch adversarial loss: 0.513465
    epoch 30; iter: 0; batch classifier loss: 27.497833; batch adversarial loss: 0.508107
    epoch 31; iter: 0; batch classifier loss: 17.455698; batch adversarial loss: 0.497301
    epoch 32; iter: 0; batch classifier loss: 27.684423; batch adversarial loss: 0.505463
    epoch 33; iter: 0; batch classifier loss: 22.626602; batch adversarial loss: 0.538608
    epoch 34; iter: 0; batch classifier loss: 25.817608; batch adversarial loss: 0.550631
    epoch 35; iter: 0; batch classifier loss: 22.358246; batch adversarial loss: 0.521186
    epoch 36; iter: 0; batch classifier loss: 15.621159; batch adversarial loss: 0.501521
    epoch 37; iter: 0; batch classifier loss: 20.917320; batch adversarial loss: 0.489194
    epoch 38; iter: 0; batch classifier loss: 16.574059; batch adversarial loss: 0.565671
    epoch 39; iter: 0; batch classifier loss: 24.028839; batch adversarial loss: 0.543856
    epoch 40; iter: 0; batch classifier loss: 21.634201; batch adversarial loss: 0.579315
    epoch 41; iter: 0; batch classifier loss: 13.560253; batch adversarial loss: 0.515843
    epoch 42; iter: 0; batch classifier loss: 22.737127; batch adversarial loss: 0.498395
    epoch 43; iter: 0; batch classifier loss: 18.391884; batch adversarial loss: 0.470165
    epoch 44; iter: 0; batch classifier loss: 19.728931; batch adversarial loss: 0.586774
    epoch 45; iter: 0; batch classifier loss: 20.208292; batch adversarial loss: 0.499542
    epoch 46; iter: 0; batch classifier loss: 16.870979; batch adversarial loss: 0.444735
    epoch 47; iter: 0; batch classifier loss: 9.900229; batch adversarial loss: 0.472292
    epoch 48; iter: 0; batch classifier loss: 12.657471; batch adversarial loss: 0.500682
    epoch 49; iter: 0; batch classifier loss: 12.617392; batch adversarial loss: 0.606505



#### Mean differences on debiased German Dataset


    Training set: Difference in mean outcomes between unprivileged and privileged groups = 0.000000
    Test set: Difference in mean outcomes between unprivileged and privileged groups = 0.000000



#### German Dataset Debiasing training classification metrics


    Classification accuracy = 0.700000
    Balanced classification accuracy = 0.500000
    Statistical parity difference = 0.000000
    Disparate impact = 1.000000
    Equal opportunity difference = 0.000000
    Average odds difference = 0.000000
    Theil_index = 0.057550



#### German Dataset Debiasing test classification metrics


    Classification accuracy = 0.700000
    Balanced classification accuracy = 0.500000
    Statistical parity difference = 0.000000
    Disparate impact = 1.000000
    Equal opportunity difference = 0.000000
    Average odds difference = 0.000000
    Theil_index = 0.057550


The results are the exact same again in our debiased model between both our training and testing datasets. This is extremely unexpected and requires some digging.


```python
print("Training original dataset labels mean = ", dataset_orig_train.labels.mean())
print("Test original dataset labels mean = ", dataset_orig_test.labels.mean())


print("Training undebiased labels mean = ", german_undebiased_train_pred.labels.mean())
print("Test undebiased labels mean = ", german_undebiased_test_pred.labels.mean())

print("Training debiased labels mean = ", german_debiased_train_pred.labels.mean())
print("Test debiased labels mean = ", german_debiased_test_pred.labels.mean())
```

    Training original dataset labels mean =  1.3
    Test original dataset labels mean =  1.3
    Training undebiased labels mean =  1.0
    Test undebiased labels mean =  1.0
    Training debiased labels mean =  1.0
    Test debiased labels mean =  1.0


We can now see our problem. Based on our training and test split and with the "shuffle" option set to True, there is exactly a 70% rate of favorable outcomes in both training and test data. Both models default to predicting favorable outcomes for every row which is why we get these seemingly perfectly unbiased predictions.

Moving on, lets try Reject Option Classification on these debiased outcomes to see if this has any improvement on this current dataset.


```python
German_ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
German_ROC = German_ROC.fit(dataset_transf_train, german_debiased_train_pred)

print("German Dataset Train: Optimal classification threshold = %.4f" % German_ROC.classification_threshold)

german_roc_train_pred = German_ROC.predict(german_debiased_train_pred)
german_roc_test_pred = German_ROC.predict(german_debiased_test_pred)

display(Markdown("#### German ROC training classification metrics"))
print_classification_metrics(german_debiased_train_pred, german_roc_train_pred)

display(Markdown("#### German ROC testing classification metrics"))
print_classification_metrics(german_debiased_test_pred, german_roc_test_pred)
```

    German Dataset Train: Optimal classification threshold = 0.9900



#### German ROC training classification metrics


    Classification accuracy = 0.972669
    Balanced classification accuracy = nan
    Statistical parity difference = -0.008218
    Disparate impact = 0.991562
    Equal opportunity difference = -0.008218
    Average odds difference = nan
    Theil_index = 0.027518


    /usr/local/lib/python3.6/dist-packages/aif360/metrics/classification_metric.py:278: RuntimeWarning: invalid value encountered in double_scalars
      TPR=TP / P, TNR=TN / N, FPR=FP / N, FNR=FN / P,
    /usr/local/lib/python3.6/dist-packages/aif360/metrics/classification_metric.py:279: RuntimeWarning: invalid value encountered in double_scalars
      GTPR=GTP / P, GTNR=GTN / N, GFPR=GFP / N, GFNR=GFN / P,



#### German ROC testing classification metrics


    Classification accuracy = 0.983297
    Balanced classification accuracy = nan
    Statistical parity difference = -0.012837
    Disparate impact = 0.986965
    Equal opportunity difference = -0.012837
    Average odds difference = nan
    Theil_index = 0.016807


Ironically, we are now actually less fair. We notice the statistical parity and equal opportunity differences are now slightly skewed in favor of the privileged group. We also notice that two fields print "nan". There are no true negatives or false positives so this means invalid values for metrics relying on these ratios.

Just for fun, we decided we would like to also try this pipeline on the Compas dataset. The Compas dataset predicts a criminal's likelihood of re-committing a crime. Given the recent controversey surrounding racism in America, we thought it would be very interesting to observe the bias impact of race in our pipeline to try and improve fairness.

### **Compas Dataset Analysis**




```python
urllib.request.urlretrieve ("https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv", "/usr/local/lib/python3.6/dist-packages/aif360/data/raw/compas/compas-scores-two-years.csv")
compas_dataset_orig = load_preproc_data_compas(['race'])
compas_dataset_orig_train, compas_dataset_orig_test = compas_dataset_orig.split([0.7], shuffle=True)
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

compas_metric_orig_train = BinaryLabelDatasetMetric(compas_dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Compas Original training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % compas_metric_orig_train.mean_difference())

compas_metric_orig_test = BinaryLabelDatasetMetric(compas_dataset_orig_test, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
display(Markdown("#### Compas Original testing dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % compas_metric_orig_test.mean_difference())

Compas_RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
compas_dataset_transf_train = Compas_RW.fit_transform(compas_dataset_orig_train)
compas_dataset_transf_test = Compas_RW.fit_transform(compas_dataset_orig_test)

compas_metric_transf_train = BinaryLabelDatasetMetric(compas_dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
display(Markdown("#### Compas Transformed training dataset"))
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % compas_metric_transf_train.mean_difference())

display(Markdown("#### Mean differences on original  compas dataset"))
print_binary_label_metrics(compas_dataset_orig_train, compas_dataset_orig_test)

display(Markdown("#### Mean differences on reweighted compas dataset"))
print_binary_label_metrics(compas_dataset_transf_train, compas_dataset_transf_test)

```


#### Compas Original training dataset


    Difference in mean outcomes between unprivileged and privileged groups = -0.147152



#### Compas Original testing dataset


    Difference in mean outcomes between unprivileged and privileged groups = -0.098170



#### Compas Transformed training dataset


    Difference in mean outcomes between unprivileged and privileged groups = 0.000000



#### Mean differences on original  compas dataset


    Training set: Difference in mean outcomes between unprivileged and privileged groups = -0.147152
    Test set: Difference in mean outcomes between unprivileged and privileged groups = -0.098170



#### Mean differences on reweighted compas dataset


    Training set: Difference in mean outcomes between unprivileged and privileged groups = 0.000000
    Test set: Difference in mean outcomes between unprivileged and privileged groups = 0.000000


Here we see a much more equitable split of bias between our training and test datasets. There is a bias of about 15% and 10% respectively for our training and test datasets in favor of the privileged group. After reweighting the dataset we get a much more equitable amount of bias between both groups. Now, let's see how our transformed datasets work on an undebiased adversarial model.


```python
sess = tf.Session()
compas_undebiased_model = AdversarialDebiasing(privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups, scope_name='plain_classifier', debias=False, sess=sess)
compas_undebiased_model.fit(compas_dataset_transf_train)
compas_undebiased_train_pred = compas_undebiased_model.predict(compas_dataset_transf_train) 
compas_undebiased_test_pred = compas_undebiased_model.predict(compas_dataset_transf_test)
display(Markdown("#### Mean differences on undebiased dataset"))
print_binary_label_metrics(compas_undebiased_train_pred, compas_undebiased_test_pred)


display(Markdown("#### Compas Undebiased training classification metrics"))
print_classification_metrics(compas_dataset_transf_train, compas_undebiased_train_pred)

display(Markdown("#### Compas Undebiased test classification metrics"))
print_classification_metrics(compas_dataset_transf_test, compas_undebiased_test_pred)

```

    epoch 0; iter: 0; batch classifier loss: 0.709799
    epoch 1; iter: 0; batch classifier loss: 0.661800
    epoch 2; iter: 0; batch classifier loss: 0.632821
    epoch 3; iter: 0; batch classifier loss: 0.605321
    epoch 4; iter: 0; batch classifier loss: 0.686599
    epoch 5; iter: 0; batch classifier loss: 0.614292
    epoch 6; iter: 0; batch classifier loss: 0.600114
    epoch 7; iter: 0; batch classifier loss: 0.605269
    epoch 8; iter: 0; batch classifier loss: 0.567148
    epoch 9; iter: 0; batch classifier loss: 0.649088
    epoch 10; iter: 0; batch classifier loss: 0.612801
    epoch 11; iter: 0; batch classifier loss: 0.624373
    epoch 12; iter: 0; batch classifier loss: 0.629660
    epoch 13; iter: 0; batch classifier loss: 0.651005
    epoch 14; iter: 0; batch classifier loss: 0.605420
    epoch 15; iter: 0; batch classifier loss: 0.571757
    epoch 16; iter: 0; batch classifier loss: 0.662565
    epoch 17; iter: 0; batch classifier loss: 0.641205
    epoch 18; iter: 0; batch classifier loss: 0.601421
    epoch 19; iter: 0; batch classifier loss: 0.579058
    epoch 20; iter: 0; batch classifier loss: 0.585631
    epoch 21; iter: 0; batch classifier loss: 0.647303
    epoch 22; iter: 0; batch classifier loss: 0.624898
    epoch 23; iter: 0; batch classifier loss: 0.656540
    epoch 24; iter: 0; batch classifier loss: 0.666094
    epoch 25; iter: 0; batch classifier loss: 0.572246
    epoch 26; iter: 0; batch classifier loss: 0.623661
    epoch 27; iter: 0; batch classifier loss: 0.630078
    epoch 28; iter: 0; batch classifier loss: 0.626637
    epoch 29; iter: 0; batch classifier loss: 0.591665
    epoch 30; iter: 0; batch classifier loss: 0.588048
    epoch 31; iter: 0; batch classifier loss: 0.589131
    epoch 32; iter: 0; batch classifier loss: 0.655511
    epoch 33; iter: 0; batch classifier loss: 0.619494
    epoch 34; iter: 0; batch classifier loss: 0.623973
    epoch 35; iter: 0; batch classifier loss: 0.647728
    epoch 36; iter: 0; batch classifier loss: 0.610190
    epoch 37; iter: 0; batch classifier loss: 0.641432
    epoch 38; iter: 0; batch classifier loss: 0.600211
    epoch 39; iter: 0; batch classifier loss: 0.623828
    epoch 40; iter: 0; batch classifier loss: 0.570642
    epoch 41; iter: 0; batch classifier loss: 0.609827
    epoch 42; iter: 0; batch classifier loss: 0.600307
    epoch 43; iter: 0; batch classifier loss: 0.596266
    epoch 44; iter: 0; batch classifier loss: 0.586958
    epoch 45; iter: 0; batch classifier loss: 0.658044
    epoch 46; iter: 0; batch classifier loss: 0.624747
    epoch 47; iter: 0; batch classifier loss: 0.563967
    epoch 48; iter: 0; batch classifier loss: 0.619611
    epoch 49; iter: 0; batch classifier loss: 0.598399



#### Mean differences on undebiased dataset


    Training set: Difference in mean outcomes between unprivileged and privileged groups = -0.286749
    Test set: Difference in mean outcomes between unprivileged and privileged groups = -0.263062



#### Compas Undebiased training classification metrics


    Classification accuracy = 0.652080
    Balanced classification accuracy = 0.649831
    Statistical parity difference = -0.286749
    Disparate impact = 0.605566
    Equal opportunity difference = -0.268766
    Average odds difference = -0.287616
    Theil_index = 0.210318



#### Compas Undebiased test classification metrics


    Classification accuracy = 0.647825
    Balanced classification accuracy = 0.644367
    Statistical parity difference = -0.263062
    Disparate impact = 0.630762
    Equal opportunity difference = -0.206288
    Average odds difference = -0.268653
    Theil_index = 0.233372


From a biased standpoint, these metrics are pretty bad and indicate high bias in favor of the privileged group. Similar values for our statistical praity, equal opportunity, and average odds all highlight a strong likelihood of favorable outcome for caucasians in our model. Our low disparate impact also lends support. Compared to our last dataset, we have different values between our training and test datasets proving that we have a more randomized split. Now, let's try debiasing our adversarial debiased model.


```python
sess.close()
tf.reset_default_graph()
sess = tf.Session()

compas_debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups, scope_name='debiased_classifier', debias=True, sess=sess)
compas_debiased_model.fit(compas_dataset_transf_train)
compas_debiased_train_pred = compas_debiased_model.predict(compas_dataset_transf_train) 
compas_debiased_test_pred = compas_debiased_model.predict(compas_dataset_transf_test)
display(Markdown("#### Mean differences on debiased Compas dataset"))
print_binary_label_metrics(compas_debiased_train_pred, compas_debiased_test_pred)


display(Markdown("#### Compas Debiasing training classification metrics"))
print_classification_metrics(compas_dataset_transf_train, compas_debiased_train_pred)

display(Markdown("#### Compas Debiasing test classification metrics"))
print_classification_metrics(compas_dataset_transf_test, compas_debiased_test_pred)

```

    epoch 0; iter: 0; batch classifier loss: 0.677774; batch adversarial loss: 0.716305
    epoch 1; iter: 0; batch classifier loss: 0.628508; batch adversarial loss: 0.715566
    epoch 2; iter: 0; batch classifier loss: 0.625697; batch adversarial loss: 0.690304
    epoch 3; iter: 0; batch classifier loss: 0.615126; batch adversarial loss: 0.676155
    epoch 4; iter: 0; batch classifier loss: 0.574680; batch adversarial loss: 0.634857
    epoch 5; iter: 0; batch classifier loss: 0.598757; batch adversarial loss: 0.678938
    epoch 6; iter: 0; batch classifier loss: 0.665726; batch adversarial loss: 0.705665
    epoch 7; iter: 0; batch classifier loss: 0.583390; batch adversarial loss: 0.686738
    epoch 8; iter: 0; batch classifier loss: 0.631026; batch adversarial loss: 0.667654
    epoch 9; iter: 0; batch classifier loss: 0.604111; batch adversarial loss: 0.659266
    epoch 10; iter: 0; batch classifier loss: 0.640399; batch adversarial loss: 0.668405
    epoch 11; iter: 0; batch classifier loss: 0.611143; batch adversarial loss: 0.655809
    epoch 12; iter: 0; batch classifier loss: 0.650580; batch adversarial loss: 0.658938
    epoch 13; iter: 0; batch classifier loss: 0.600530; batch adversarial loss: 0.678144
    epoch 14; iter: 0; batch classifier loss: 0.609655; batch adversarial loss: 0.651139
    epoch 15; iter: 0; batch classifier loss: 0.621671; batch adversarial loss: 0.669958
    epoch 16; iter: 0; batch classifier loss: 0.599420; batch adversarial loss: 0.663548
    epoch 17; iter: 0; batch classifier loss: 0.567450; batch adversarial loss: 0.651917
    epoch 18; iter: 0; batch classifier loss: 0.710643; batch adversarial loss: 0.669632
    epoch 19; iter: 0; batch classifier loss: 0.607392; batch adversarial loss: 0.635241
    epoch 20; iter: 0; batch classifier loss: 0.605862; batch adversarial loss: 0.660496
    epoch 21; iter: 0; batch classifier loss: 0.603220; batch adversarial loss: 0.645372
    epoch 22; iter: 0; batch classifier loss: 0.592495; batch adversarial loss: 0.650416
    epoch 23; iter: 0; batch classifier loss: 0.612319; batch adversarial loss: 0.637218
    epoch 24; iter: 0; batch classifier loss: 0.606113; batch adversarial loss: 0.637277
    epoch 25; iter: 0; batch classifier loss: 0.609981; batch adversarial loss: 0.655257
    epoch 26; iter: 0; batch classifier loss: 0.577645; batch adversarial loss: 0.665574
    epoch 27; iter: 0; batch classifier loss: 0.660513; batch adversarial loss: 0.643912
    epoch 28; iter: 0; batch classifier loss: 0.579243; batch adversarial loss: 0.639707
    epoch 29; iter: 0; batch classifier loss: 0.627600; batch adversarial loss: 0.632814
    epoch 30; iter: 0; batch classifier loss: 0.643466; batch adversarial loss: 0.657322
    epoch 31; iter: 0; batch classifier loss: 0.593191; batch adversarial loss: 0.642938
    epoch 32; iter: 0; batch classifier loss: 0.630770; batch adversarial loss: 0.656479
    epoch 33; iter: 0; batch classifier loss: 0.596158; batch adversarial loss: 0.662204
    epoch 34; iter: 0; batch classifier loss: 0.641588; batch adversarial loss: 0.650079
    epoch 35; iter: 0; batch classifier loss: 0.599647; batch adversarial loss: 0.630278
    epoch 36; iter: 0; batch classifier loss: 0.584255; batch adversarial loss: 0.644303
    epoch 37; iter: 0; batch classifier loss: 0.663150; batch adversarial loss: 0.650094
    epoch 38; iter: 0; batch classifier loss: 0.627804; batch adversarial loss: 0.679217
    epoch 39; iter: 0; batch classifier loss: 0.634585; batch adversarial loss: 0.665857
    epoch 40; iter: 0; batch classifier loss: 0.684100; batch adversarial loss: 0.681380
    epoch 41; iter: 0; batch classifier loss: 0.627548; batch adversarial loss: 0.652162
    epoch 42; iter: 0; batch classifier loss: 0.592801; batch adversarial loss: 0.624520
    epoch 43; iter: 0; batch classifier loss: 0.573675; batch adversarial loss: 0.627429
    epoch 44; iter: 0; batch classifier loss: 0.605757; batch adversarial loss: 0.663803
    epoch 45; iter: 0; batch classifier loss: 0.597731; batch adversarial loss: 0.673770
    epoch 46; iter: 0; batch classifier loss: 0.659634; batch adversarial loss: 0.673271
    epoch 47; iter: 0; batch classifier loss: 0.722117; batch adversarial loss: 0.653565
    epoch 48; iter: 0; batch classifier loss: 0.571397; batch adversarial loss: 0.638071
    epoch 49; iter: 0; batch classifier loss: 0.636164; batch adversarial loss: 0.683166



#### Mean differences on debiased Compas dataset


    Training set: Difference in mean outcomes between unprivileged and privileged groups = 0.079449
    Test set: Difference in mean outcomes between unprivileged and privileged groups = 0.096882



#### Compas Debiasing training classification metrics


    Classification accuracy = 0.656641
    Balanced classification accuracy = 0.656085
    Statistical parity difference = 0.079449
    Disparate impact = 1.168391
    Equal opportunity difference = 0.090928
    Average odds difference = 0.078896
    Theil_index = 0.244396



#### Compas Debiasing test classification metrics


    Classification accuracy = 0.658337
    Balanced classification accuracy = 0.656715
    Statistical parity difference = 0.096882
    Disparate impact = 1.204878
    Equal opportunity difference = 0.145139
    Average odds difference = 0.092130
    Theil_index = 0.248094


These are great results! A disparate impact above 1 and positive values for all other key bias metrics indicate more favorable outcomes for non-caucasians in our model. These are a marked improvement over the negative indicators in our undebiased model so we can clearly see that this model is working very effectively. Also worth noting, we are improving fairness while not trading off with accuracy. Next, let's test our model on the original dataset to see if our reweighted pre-processing is actually working or if the step is unnecessary. 


```python
sess.close()
tf.reset_default_graph()
sess = tf.Session()

compas_debiased_model_no_preproc = AdversarialDebiasing(privileged_groups = privileged_groups, unprivileged_groups = unprivileged_groups, scope_name='debiased_classifier', debias=True, sess=sess)
compas_debiased_model_no_preproc.fit(compas_dataset_orig_train)
compas_debiased_train_pred_no_preproc = compas_debiased_model_no_preproc.predict(compas_dataset_orig_train) 
compas_debiased_test_pred_no_preproc = compas_debiased_model_no_preproc.predict(compas_dataset_orig_test)
display(Markdown("#### Mean differences on debiased Compas dataset WITHOUT pre-processing"))
print_binary_label_metrics(compas_debiased_train_pred_no_preproc, compas_debiased_test_pred_no_preproc)


display(Markdown("#### Compas Debiasing training classification metrics WITHOUT pre-processing"))
print_classification_metrics(compas_dataset_orig_train, compas_debiased_train_pred_no_preproc)

display(Markdown("#### Compas Debiasing test classification metrics WITHOUT pre-processing"))
print_classification_metrics(compas_dataset_orig_test, compas_debiased_test_pred_no_preproc)

```

    epoch 0; iter: 0; batch classifier loss: 0.718103; batch adversarial loss: 0.835899
    epoch 1; iter: 0; batch classifier loss: 0.665195; batch adversarial loss: 0.830785
    epoch 2; iter: 0; batch classifier loss: 0.697499; batch adversarial loss: 0.840313
    epoch 3; iter: 0; batch classifier loss: 0.654141; batch adversarial loss: 0.766697
    epoch 4; iter: 0; batch classifier loss: 0.613985; batch adversarial loss: 0.827360
    epoch 5; iter: 0; batch classifier loss: 0.752280; batch adversarial loss: 0.732054
    epoch 6; iter: 0; batch classifier loss: 0.661658; batch adversarial loss: 0.754647
    epoch 7; iter: 0; batch classifier loss: 0.699477; batch adversarial loss: 0.747449
    epoch 8; iter: 0; batch classifier loss: 0.579388; batch adversarial loss: 0.772329
    epoch 9; iter: 0; batch classifier loss: 0.660209; batch adversarial loss: 0.729917
    epoch 10; iter: 0; batch classifier loss: 0.617608; batch adversarial loss: 0.710395
    epoch 11; iter: 0; batch classifier loss: 0.626631; batch adversarial loss: 0.742258
    epoch 12; iter: 0; batch classifier loss: 0.619737; batch adversarial loss: 0.720179
    epoch 13; iter: 0; batch classifier loss: 0.636193; batch adversarial loss: 0.715852
    epoch 14; iter: 0; batch classifier loss: 0.669188; batch adversarial loss: 0.695140
    epoch 15; iter: 0; batch classifier loss: 0.639255; batch adversarial loss: 0.690818
    epoch 16; iter: 0; batch classifier loss: 0.626940; batch adversarial loss: 0.684189
    epoch 17; iter: 0; batch classifier loss: 0.613016; batch adversarial loss: 0.682489
    epoch 18; iter: 0; batch classifier loss: 0.628490; batch adversarial loss: 0.664535
    epoch 19; iter: 0; batch classifier loss: 0.656190; batch adversarial loss: 0.691512
    epoch 20; iter: 0; batch classifier loss: 0.586034; batch adversarial loss: 0.673141
    epoch 21; iter: 0; batch classifier loss: 0.608783; batch adversarial loss: 0.673675
    epoch 22; iter: 0; batch classifier loss: 0.566546; batch adversarial loss: 0.687173
    epoch 23; iter: 0; batch classifier loss: 0.609590; batch adversarial loss: 0.669297
    epoch 24; iter: 0; batch classifier loss: 0.631112; batch adversarial loss: 0.638361
    epoch 25; iter: 0; batch classifier loss: 0.636683; batch adversarial loss: 0.659536
    epoch 26; iter: 0; batch classifier loss: 0.586456; batch adversarial loss: 0.661069
    epoch 27; iter: 0; batch classifier loss: 0.659256; batch adversarial loss: 0.666094
    epoch 28; iter: 0; batch classifier loss: 0.636565; batch adversarial loss: 0.657279
    epoch 29; iter: 0; batch classifier loss: 0.564716; batch adversarial loss: 0.674474
    epoch 30; iter: 0; batch classifier loss: 0.599840; batch adversarial loss: 0.638653
    epoch 31; iter: 0; batch classifier loss: 0.609629; batch adversarial loss: 0.671869
    epoch 32; iter: 0; batch classifier loss: 0.621630; batch adversarial loss: 0.675226
    epoch 33; iter: 0; batch classifier loss: 0.594112; batch adversarial loss: 0.670461
    epoch 34; iter: 0; batch classifier loss: 0.606289; batch adversarial loss: 0.654465
    epoch 35; iter: 0; batch classifier loss: 0.608102; batch adversarial loss: 0.688618
    epoch 36; iter: 0; batch classifier loss: 0.649477; batch adversarial loss: 0.688615
    epoch 37; iter: 0; batch classifier loss: 0.595800; batch adversarial loss: 0.646859
    epoch 38; iter: 0; batch classifier loss: 0.662301; batch adversarial loss: 0.645705
    epoch 39; iter: 0; batch classifier loss: 0.633352; batch adversarial loss: 0.638589
    epoch 40; iter: 0; batch classifier loss: 0.635687; batch adversarial loss: 0.672884
    epoch 41; iter: 0; batch classifier loss: 0.599615; batch adversarial loss: 0.669994
    epoch 42; iter: 0; batch classifier loss: 0.595701; batch adversarial loss: 0.635616
    epoch 43; iter: 0; batch classifier loss: 0.592493; batch adversarial loss: 0.653228
    epoch 44; iter: 0; batch classifier loss: 0.625116; batch adversarial loss: 0.661298
    epoch 45; iter: 0; batch classifier loss: 0.632405; batch adversarial loss: 0.633305
    epoch 46; iter: 0; batch classifier loss: 0.566538; batch adversarial loss: 0.664223
    epoch 47; iter: 0; batch classifier loss: 0.662686; batch adversarial loss: 0.633110
    epoch 48; iter: 0; batch classifier loss: 0.624095; batch adversarial loss: 0.633043
    epoch 49; iter: 0; batch classifier loss: 0.700290; batch adversarial loss: 0.661171



#### Mean differences on debiased Compas dataset WITHOUT pre-processing


    Training set: Difference in mean outcomes between unprivileged and privileged groups = -0.226713
    Test set: Difference in mean outcomes between unprivileged and privileged groups = -0.187631



#### Compas Debiasing training classification metrics WITHOUT pre-processing


    Classification accuracy = 0.671630
    Balanced classification accuracy = 0.667824
    Statistical parity difference = -0.226713
    Disparate impact = 0.687640
    Equal opportunity difference = -0.159519
    Average odds difference = -0.183673
    Theil_index = 0.191776



#### Compas Debiasing test classification metrics WITHOUT pre-processing


    Classification accuracy = 0.667298
    Balanced classification accuracy = 0.660438
    Statistical parity difference = -0.187631
    Disparate impact = 0.734117
    Equal opportunity difference = -0.117409
    Average odds difference = -0.163888
    Theil_index = 0.205933


These results are a clear step in the wrong direction. The negative values for the difference statistics imply a lack of parity between privileged and unprivileged group so while adversarial debiasing on the original dataset offers a slight improvement in fairness compared to the undebiased model it is apparent that pre-processing is key to achieving desirable results. While we have obtained very fair results with reweighting and adversarial debiasing, our accuracy hovers around 65% which isn't great. Next, let's look at rejection option classification to try to maintain fairness while also optimizing accuracy.


```python
Compas_ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
Compas_ROC = Compas_ROC.fit(compas_dataset_transf_train, compas_debiased_train_pred)

print("Train: Optimal classification threshold = %.4f" % Compas_ROC.classification_threshold)

compas_roc_train_pred = Compas_ROC.predict(compas_debiased_train_pred)
compas_roc_test_pred = Compas_ROC.predict(compas_debiased_test_pred)

display(Markdown("#### Compas ROC training classification metrics"))
print_classification_metrics(compas_debiased_train_pred, compas_roc_train_pred)

display(Markdown("#### Compas ROC testing classification metrics"))
print_classification_metrics(compas_debiased_test_pred, compas_roc_test_pred)
```

    Train: Optimal classification threshold = 0.4951



#### Compas ROC training classification metrics


    Classification accuracy = 0.983360
    Balanced classification accuracy = 0.982694
    Statistical parity difference = 0.038141
    Disparate impact = 1.074330
    Equal opportunity difference = 0.000000
    Average odds difference = -0.039104
    Theil_index = 0.006142



#### Compas ROC testing classification metrics


    Classification accuracy = 0.985904
    Balanced classification accuracy = 0.984935
    Statistical parity difference = 0.060576
    Disparate impact = 1.118966
    Equal opportunity difference = 0.000000
    Average odds difference = -0.034438
    Theil_index = 0.005197


Reject option classifiers are supposed to swap predictions in favor of the unprivileged group when predictions are near the decision boundary. Ironically, doing this actually reduced bias in favor of the unprivileged group compared to the model we built earlier. Does that mean our results are worse? No, I don't think so because there is a marked improvement in accuracy. We built a classifier that is both very accurate (Well above 0.98 accuracy) and has no evidence of bias per the key bias metrics. In summarization, our pipeline offers the best of both worlds by getting very fair models to mitigate bias while not sacrificing accuracy.

### **Conclusion**

We created three stages in our fairnes pipeline and below we discuss their overall performance on both the Compas and German Datasets.

**Reweighting**

In pre-processing we modified the weights of our features to try and create more favorable outcomes for the unprivileged group. In both datasets we saw the mean difference in outcomes between privileged and unprivileged reduced to 0 so we now have a more fair dataset to train a model.

**Adversarial Debiasing**

In in-processing we trained a model to make it intentionally difficult to try and guess the protected attribute based on the outcome. By trying to remove the correlation between the protected attribute and the outcome we noticed a significant swing in the opposite direction and noticed a lot of bias in favor of the unprivilieged group in the Compas dataset. However, accuracy wasn't great so additional work is needed to create a desirable classification.

In the German dataset we didn't notice anything particularly interesting. Because of the split in the training and testing data our model just predicted favorable outcomes for everyone so it wasn't a great model.

**Reject Option Classification**

In post-processing we try to flip the predictions very near to our decision boundary to generate more fair predictions than those that our model has made.

In the Compas dataset, this had the ironic effect of generating more outcomes in favor of the privileged group but doing so actually made our predictions more fair because the unprivileged group received much more favorable outcomes from our model. The biggest improvement in this stage was accuracy which jumped from about 65% to over 98%. Our pipeline worked extremely well on the Compas dataset providing excellent fairness metrics and excellent accuracy metrics.

In the German dataset, this stage also resulted in slightly more favorable outcomes for the privileged group but still provided great fairness metrics. Our model generated predictions at 70% accuracy which isn't great but by switching predictions near the decision boundary we were able to get our accuracy up to about 98%. Similar to the Compas dataset, we compromised fairness in a few instances but were able to get much more accuracte predictions but we think this trade-off ultimately provides a greater model.

