# Applying Lu(23)'s method on Asian options

Pricing **Asian options of a basket of 𝑛 stocks** using Lu(23)'s method is documented in Section 6 of my report.

The codes are in path "/Asian options/"

## How to test
#### For "*Asian options of a basket of 100 stocks*", testing files involve:
- `unit_test_asian100.py`: for the unit test;
- `integration_test_asian100.py`: for the integration test; and
- `regression_test_asian100.py`: for the regression test

In command line, run 
```python
  python -m unittest unit_test_asian100; # for unit test
  python -m unittest integration_test_asian100; # for integration test
  python -m unittest regression_test_asian100; # for regression test
```


Trained models and MC price are included in path `./Weights_Asian/`. 


This is an **example** of running `integration_test_asian100.py`:
```python
$ python -m unittest integration_test_asian100    
Model Loaded!
Predicting...

********************************************************************************
The fitted value is 11.586200544285191
The MC value is 11.569540649040118
The relative error is 0.0014399789715467742

Under the error tolerence in the order of O(10^(-3)), we consider the fitted value is the same with the solution to Asian Basket Option
.
----------------------------------------------------------------------
Ran 1 test in 19.798s

OK
```

This is an **example** of running `unit_test_asian100.py`:
```python
$ python -m unittest unit_test_asian100
.............
----------------------------------------------------------------------
Ran 13 tests in 1.302s

OK
```

## To train from scratch
If you want to train the model from scratch, you can find a dedicated script for each experiment in report's Section 6. 
- `script_asian_basket_option_arith.py`: my application of Lu(23)'s method on **a basket Asian options on 100 assets**, with Arithmetic average, 
- `script_asian_basket_option_geo.py`: my application of Lu(23)'s method on **a basket Asian options on 100 assets**, with Geometric average
- You can change `dim = 100` in the script to another dimension such as `dim = 1, 5, 10, 50`.

In command line, run
```python
  python script_asian_basket_option_arith.py
```
```python
  python script_asian_basket_option_geo.py
```

For Deep solver method, this part of code is modified from (https://github.com/AlessandroGnoatto/DeepBsdeSolverWithJumps). 
<br/>
```python
  python script_asian_basket_option_avg_deepsolver.py
```
<br/>

The results we obtained and documented in the final report are also in <https://drive.google.com/drive/folders/1IA8Q2vOkdJKobUJM4M0PMPXxXaitD6T3?usp=drive_link.>

