# FX rate forecasting using LSTM with News indices.
A model that predicts FX rate using News indices.  Note: You'll need to download news data before running it.  
https://marketpsych-website.s3.amazonaws.com/web3/files/papers/Forecasting%20the%20USD-JPY%20Rate%20with%20Sentiment.pdf

* USDJPY rate and Price prediction from news (-1.0*rma).

![image1](https://github.com/knkasa/LSTM_forecast/blob/main/USDJPY%20priceDirection(JPY).png)

* Earned profit(pips) based on the forward test using the above model.  Buy position if the model predicts increase in price, and sell if the prediction price drops.  (Note: the actuall pips is 100 times that of the value in the graph.  So, roughly 2500 pips is earned between the year 2019~2022)

![image2](https://github.com/knkasa/LSTM_forecast/blob/main/Earned%20profit%20from%20forward%20test.png)

* Feature importance using sarpley. (rate_maxmin=MaxMin standardized USDJPY rate.  pricePrediction=News prediction.  macd_diff=MACD of pricePrediction.  US,JP=USA and Japan. Vol=Volatility)

![image3](https://github.com/knkasa/LSTM_forecast/blob/main/Variable%20importance2.png)
