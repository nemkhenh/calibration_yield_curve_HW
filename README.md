# Calibration yield curve with 2F-Vasicek and Hull-white extension
- The calibration attemps to understand the intricacies of interest rate dynamics by evaluating the effectiveness of 3 models: 2F Vasicek, 1F Hull-White, 2F Hull-White


# Why this comparison ?
One of the main challenges in interest rate modeling is the risk taht the model may not accurately capture the true dynamics of interest rates. The goal for this analysis is to demonstrate that 1F,2F HW and 2F Vasicek models demonstrate how the additional variability of volatility and mean reversion of the 2F models give a better picture of the evolution of interest rates

# Brief conception
Stochastic models have probabilistics elements, whide deterministic models are based on certain initial parameters. These models admit that the result may vary since they also depend other random measures and are thus most fittingly expressed in terms of probability densities. About stochastic processes, which are collections of random variables that evolves through time, these provide a dynamic framework for explaining how systems change under conditions of uncertainty.

Stochastic models are used for 2 essentials:
- For the valuation and hedging of interest rate products that deliver random cash flows in the future. The option seller must be able to price the product they are selling, but above all, to replicate (or hedge) the option they are selling because they incur an unlimited loss.
- For the implementation of scenario analysis. With the help of stochastic processes, various situations for financial variables can be created, thus providing more input in managing risks in practical financial decisison making.

An ideal interest model is a model that is:
1) realistic in the sense that it allows us to take into account the empirical properties of the yield curve highlighted previously.
2) well constructed in the senses that the model are observable in the market or easily estimable, and moreover frequently readjustable.
3) compatible with market prices of plain vanilla products.

The use of actual data for calibrating these models is considerably more intricate and prone to several obstacles than when using simulated data, largely because of the influence of market volatility, economic events and data quality factors. Thus, it is important to focus on the reserach that would hemp apply theoretical models developed to the actual work. 

This project aims to contributes to both theoretical understanding and practical calibration of interest models by using actual data. Data source used comes from US Department of the Treasury and Bloomberg.
  
# Vasicek model
