# Lu(23)'s implementations

The implementation of Lu(23)'s methodology and results is documented in Section 5 of my report.  
The codes are in path "./Lu/"

## How to test
#### For Lu(23)'s "*One dimensional pure jump process*", testing files involve:
- `unit_test_onepure.py`: for the unit test;
- `integration_test_onepure.py`: for the integrated test; and
- `regression_test_onepure.py`: for the regression test

In command line, run 
```python
  python -m unittest unit_test_onepure; # for unit test
  python -m unittest integration_test_onepure; # for integration test
  python -m unittest regression_test_onepure; # for regression test
```
#### For Lu(23)'s "*Robustness*", testing files involve:
- `unit_test_equation2.py`
- `integration_test_robust.py`
- `regression_test_robust.py`

In command line, run 
```python
  python -m unittest unit_test_equation2; # for unit test
  python -m unittest integration_test_robust; # for integration test
  python -m unittest regression_test_robust; # for regression test
```
#### For Lu(23)'s "*High dimensional problems*", testing files involve:
- `unit_test_highdim.py`
- `integration_test_highdim.py`
- `regression_test_highdim.py`

In command line, run 
```python
  python -m unittest unit_test_highdim; # for unit test
  python -m unittest integration_test_highdim; # for integration test
  python -m unittest regression_test_highdim; # for regression test
```


Trained models are included in path `./model_zoo/`. 


This is an **example** of running `integration_test_onepure.py`:
```python
$ python -m unittest integration_test_onepure
Model Loaded!
Predicting...

********************************************************************************
The fitted value is 1.0001029828280779
Under the error tolerence in the order of O(10^(-4)), we consider the fitted value is the same with the solution to u(T,x)= x
.
----------------------------------------------------------------------
Ran 1 test in 15.474s

OK
```

This is an **example** of running `unit_test_onepure.py`:
```python
$ python -m unittest unit_test_onepure         
..............
----------------------------------------------------------------------
Ran 14 tests in 1.147s

OK
```

## To train from scratch
If you want to train the model from scratch, you can find a dedicated script for each experiment in report's Section 5. You can run:
- `script_onepure.py`: my replication of Lu(23)'s "*One dimensional pure jump process*" experiment.
- `script_robust_M_125_N_20_I_250.py`: one of the "*Robustness*" experiment on Lu(23)'s method across different numbers of trajactories M and time inverval N. Here M=125, N=20, Iteration=250.
- `script_highdim.py`: my replication of Lu(23)'s "*High dimensional problems*" experiment.
with the saved model.

In command line, run
```python
  python script_onepure.py
```
```python
  python script_robust_M_125_N_20_I_250.py
```
```python
  python script_highdim.py
```
<br/>

All the training results we obtained and documented in the final report can be found in 
  <https://drive.google.com/drive/folders/1IA8Q2vOkdJKobUJM4M0PMPXxXaitD6T3?usp=drive_link>.

## Reference
[1] Lu, L., Guo, H., Yang, X., & Zhu, Y. (2023). Temporal difference learning for high-dimensional PIDEs with jumps. arXiv preprint arXiv:2307.02766.
