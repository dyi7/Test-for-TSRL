# Applying Lu(23)'s method to various Barrier options

In Section 7 of my final report, we show the pricing of:

  - **different barrier options**, i.e., Up-and-in, Up-and-out, Down-and-in, Down-and-out, using Lu(23)'s method in Section 7.2, 

  - **up-and-out put option with different 𝑋0** in Section 7.3, and 

  - **American up-and-out put option** in Section 7.4.

The codes are in path "/Barrier options/"


## How to test
#### For the "*Up-and-out put option*", testing files involve:
- `unit_test_upoutput.py`: for the unit test;
- `integration_test_upoutput.py`: for the integration test; and
- `regression_test_upoutput.py`: for the regression test

In command line, run 
```python
  python -m unittest unit_test_upoutput; # for unit test
  python -m unittest integration_test_upoutput; # for integration test
  python -m unittest regression_test_upoutput; # for regression test
```

#### For "*American up-and-out put option*", testing files involve:
- `unit_test_upoutput_ameri.py`: for the unit test;
- `integration_test_upoutput_ameri.py`: for the integrated test; and
- `regression_test_upoutput_ameri.py`: for the regression test

In command line, run 
```python
  python -m unittest unit_test_upoutput_ameri; # for unit test
  python -m unittest integration_test_upoutput_ameri; # for integration test
  python -m unittest regression_test_upoutput_ameri; # for regression test
```

Trained models are included in path `./model_zoo/`. 


This is an **example** of running `integration_test_upoutput.py`:
```python
$ python -m unittest integration_test_upoutput
Model Loaded!
Predicting...

********************************************************************************
The fitted value is 0.1554673121467061
The MC value is 0.15595926349611894
The absolute error is 0.0004919513494128502

Under the error tolerence in the order of O(10^(-4)), we consider the fitted value is the same with the solution to Up-and-Out put option
.
----------------------------------------------------------------------
Ran 1 test in 17.368s

OK
```

This is an **example** of running `unit_test_upoutput.py`:
```python
$ python -m unittest unit_test_upoutput
.............
----------------------------------------------------------------------
Ran 13 tests in 1.280s

OK 
```

## To train from scratch
- `script_barrier_up_out_put.py`: my application of Lu(23)'s method on **up-and-out PUT option**. You can change `barrier_direction='Up'` and `optionality = 'Out'` in the script to another barrier options such as 'up-and-in', 'down-and-in', 'down-and-out'.
- `script_barrier_up_out_call.py`: my application of Lu(23)'s method on **up-and-out CALL option**.
- `script_barrier_up_out_put_s0_0.6.py`: my application of Lu(23)'s method on **up-and-out PUT option with 𝑋0=0.6**. You can change `x_init = 0.6` to other values like `x_init = 0.8, 1.0, 1.2, 1.4`.
- `script_barrier_up_out_put_ameri.py`: my application of Lu(23)'s method on **American up-and-out PUT option**.

In command line, run
```python
  python script_barrier_up_out_put.py
```
```python
  python script_barrier_up_out_put_ameri.py
```
```python
  python script_barrier_up_out_put_s0_0.6.py
```

<br/>

For Deep solver method, this part of code is modified from (https://github.com/AlessandroGnoatto/DeepBsdeSolverWithJumps). 
```python
  python script_barrier_up_out_call_deepsolver.py
```
<br/>

The results we obtained and documented in the final report are also in <https://drive.google.com/drive/folders/1IA8Q2vOkdJKobUJM4M0PMPXxXaitD6T3?usp=drive_link.>


## Reference

[1] Lu, L., Guo, H., Yang, X., & Zhu, Y. (2023). Temporal difference learning for high-dimensional PIDEs with jumps. arXiv preprint arXiv:2307.02766.
