# Applying Lu(23)'s method on Asian options

Pricing **Asian options of a basket of 𝑛 stocks** using Lu(23)'s method is documented in Section 2 of my report.


## How to run

- `script_asian_basket_option_arith.py`: my application of Lu(23)'s method on **a basket Asian options on 100 assets**, with Arithmetic average, 
- `script_asian_basket_option_geo.py`: my application of Lu(23)'s method on **a basket Asian options on 100 assets**, with Geometric average
- You can change `dim = 100` in the script to another dimension such as `dim = 1, 5, 10, 50`.

For Deep solver method, this part of code is modified from (https://github.com/AlessandroGnoatto/DeepBsdeSolverWithJumps). 
<br/>
In `script_asian_basket_option_avg_deepsolver.py`, please change `eqn_name` to `"eqn_name": "AsianBasketCallOption"`.

<br/>

The results we obtained and documented in the final report are also in <https://drive.google.com/drive/folders/1IA8Q2vOkdJKobUJM4M0PMPXxXaitD6T3?usp=drive_link.>
