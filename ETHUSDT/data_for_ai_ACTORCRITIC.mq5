//+------------------------------------------------------------------+
//|                                      data_for_ai_ACTORCRITIC.mq5 |
//|                                  Copyright 2021, MetaQuotes Ltd. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
class Cline
{
   private:
   public:
      double line[];
};

enum MA_PERIODS
{
   MA_1=0,
   MA_10,
   MA_20,
   MA_30,
   MA_50,
   MA_100,
   MA_200,
   MA_400
};

ENUM_TIMEFRAMES tf_small = PERIOD_D1;
ENUM_TIMEFRAMES tf_middle = PERIOD_H4;
ENUM_TIMEFRAMES tf_large = PERIOD_D1;

string dataSymbol = "bn_ETHUSDT";
//string dataSymbol = "bn_BTCUSDT";
//string dataSymbol = "bn_BNBUSDT";
//string dataSymbol = "fm_APPLE";
string start_date_from = "2018-01-01";
int days_in_vector_1 = 5;
int days_in_vector_2 = 5;
datetime time_1h[], time_4h[], time_1d[];
double open_1h[], open_4h[], open_1d[];
double high_1h[], high_4h[], high_1d[];
double low_1h[], low_4h[], low_1d[];
double close_1h[], close_4h[], close_1d[];

string line;
int data_file_handle;

int rsi_1h_handle,rsi_4h_handle,rsi_1d_handle;
double rsi_1h[],rsi_4h[],rsi_1d[];
int ma_1h_handles[8],ma_4h_handles[6],ma_1d_handles[6];
int ma_1h_periods[8],ma_4h_periods[6],ma_1d_periods[6]; //= { 1,    10,   20,   30,   50,   100,     200,  400};
int ma_1h_count,ma_4h_count,ma_1d_count;
Cline ma_1h_matrix[8],ma_4h_matrix[6],ma_1d_matrix[6];
int stoch_1h_handle,stoch_4h_handle,stoch_1d_handle;
Cline stoch_1h[2],stoch_4h[2], stoch_1d[2];

int aroon_1h_handle=-1, lrs_1h_handle=-1, r2_1h_handle=-1, rmo_1h_handle=-1; 
Cline aroon_1h[2];
double lrs_1h[], r2_1h[], rmo_1h[];

int percentile_1h_100_50_handle=-1,percentile_1h_100_80_handle=-1,percentile_1h_100_90_handle=-1;
double percentile_1h_100_50[], percentile_1h_100_80[], percentile_1h_100_90[];
int percentile_4h_100_50_handle=-1,percentile_4h_100_80_handle=-1,percentile_4h_100_90_handle=-1;
double percentile_4h_100_50[], percentile_4h_100_80[], percentile_4h_100_90[];


int demarker_1h_handle = -1,demarker_4h_handle = -1,demarker_1d_handle = -1;
double demarker_1h[],demarker_4h[],demarker_1d[];
int momentum_1h_handle = -1,momentum_4h_handle = -1,momentum_1d_handle = -1;
double momentum_1h[],momentum_4h[],momentum_1d[];
int wpr_1h_handle = -1,wpr_4h_handle = -1,wpr_1d_handle = -1;
double wpr_1h[],wpr_4h[],wpr_1d[];

int bb_1h_handle,bb_4h_handle;
Cline bb_1h[2],bb_4h[3];

int macd_1h_handle,macd_4h_handle,macd_1d_handle;
Cline macd_1h[2],macd_4h[2],macd_1d[2];


void OnStart()
  {
//---

   mt_ai_data_init();
   mt_ai_data_update();
   mt_ai_data_init_file();
   for (int b=iBarShift(dataSymbol,tf_small,StringToTime(start_date_from));b>=0;b--)
      {
         string r=mt_ai_data_get(b);
         mt_ai_data_save_file(r);
      }
   mt_ai_data_close_file();
   
  }
//+------------------------------------------------------------------+



void mt_ai_data_init()
{
     
   //1 HOUR
   rsi_1h_handle = iCustom(dataSymbol,tf_small,"bn_RSI",14);
   ArraySetAsSeries(rsi_1h, true);
   
   ArrayInitialize(ma_1h_handles, -1);
   ma_1h_periods[0]=1;  ma_1h_periods[1]=10;    ma_1h_periods[2]=20;    ma_1h_periods[3]=30;
   ma_1h_periods[4]=50; ma_1h_periods[5]=100;   ma_1h_periods[6]=200;   ma_1h_periods[7]=400;
   
   ma_1h_count = ArraySize(ma_1h_handles);

   for (int ma=0; ma<ma_1h_count; ma++)
   {
      ma_1h_handles[ma]=iMA(dataSymbol, tf_small, ma_1h_periods[ma], 0, MODE_EMA, PRICE_CLOSE);
      ArraySetAsSeries(ma_1h_matrix[ma].line, true);
   }
   
   stoch_1h_handle = iStochastic(dataSymbol,tf_small, 5, 3, 3,MODE_SMA,STO_LOWHIGH);
   ArraySetAsSeries(stoch_1h[0].line, true);
   ArraySetAsSeries(stoch_1h[1].line, true);
   
   bb_1h_handle = iBands(dataSymbol,tf_small, 22, 0, 1, PRICE_CLOSE);
   ArraySetAsSeries(bb_1h[0].line,true);
   ArraySetAsSeries(bb_1h[1].line,true);
   
   macd_1h_handle = iMACD(dataSymbol, tf_small, 12, 26, 9, PRICE_CLOSE);
   ArraySetAsSeries(macd_1h[0].line, true);
   ArraySetAsSeries(macd_1h[1].line, true);
   
   aroon_1h_handle = iCustom(dataSymbol, tf_small, "classifier SKLEARN\\cl_AROON", 80);
   ArraySetAsSeries(aroon_1h[0].line, true);
   ArraySetAsSeries(aroon_1h[1].line, true);
   
   lrs_1h_handle = iCustom(dataSymbol, tf_small, "classifier SKLEARN\\cl_LRS", 80);
   ArraySetAsSeries(lrs_1h, true);
   
   r2_1h_handle = iCustom(dataSymbol, tf_small, "classifier SKLEARN\\cl_r2", 80,0.6);
   ArraySetAsSeries(r2_1h, true);
   
   rmo_1h_handle = iCustom(dataSymbol, tf_small, "classifier SKLEARN\\cl_rmo_lb");
   ArraySetAsSeries(rmo_1h, true);
   
   percentile_1h_100_50_handle=iCustom(dataSymbol, tf_small, "bn_percentile", 30, 50);
   ArraySetAsSeries(percentile_1h_100_50, true);
   percentile_1h_100_80_handle=iCustom(dataSymbol, tf_small, "bn_percentile", 30, 80);
   ArraySetAsSeries(percentile_1h_100_80, true);
   percentile_1h_100_90_handle=iCustom(dataSymbol, tf_small, "bn_percentile", 30, 90);
   ArraySetAsSeries(percentile_1h_100_90, true);
   
   //4 HOUR
   rsi_4h_handle = iCustom(dataSymbol,tf_middle,"bn_RSI",14);
   ArraySetAsSeries(rsi_4h, true);
   
   ArrayInitialize(ma_4h_handles, -1);
   ma_4h_periods[0]=1;  ma_4h_periods[1]=10;    ma_4h_periods[2]=20;    ma_4h_periods[3]=30;
   ma_4h_periods[4]=50; ma_4h_periods[5]=100;
   
   ma_4h_count = ArraySize(ma_4h_handles);

   for (int ma=0; ma<ma_4h_count; ma++)
   {
      ma_4h_handles[ma]=iMA(dataSymbol, tf_middle, ma_4h_periods[ma], 0, MODE_EMA, PRICE_CLOSE);
      ArraySetAsSeries(ma_4h_matrix[ma].line, true);
   }
   
   bb_4h_handle = iBands(dataSymbol,tf_middle, 22, 0, 1, PRICE_CLOSE);
   ArraySetAsSeries(bb_4h[0].line,true);
   ArraySetAsSeries(bb_4h[1].line,true);
   ArraySetAsSeries(bb_4h[2].line,true);
   
   stoch_4h_handle = iStochastic(dataSymbol,tf_middle, 5, 3, 3,MODE_SMA,STO_LOWHIGH);
   ArraySetAsSeries(stoch_4h[0].line, true);
   ArraySetAsSeries(stoch_4h[1].line, true);
            
   macd_4h_handle = iMACD(dataSymbol, tf_middle, 12, 26, 9, PRICE_CLOSE);
   ArraySetAsSeries(macd_4h[0].line, true);
   ArraySetAsSeries(macd_4h[1].line, true);
   
   percentile_4h_100_50_handle=iCustom(dataSymbol, tf_middle, "bn_percentile", 30, 50);
   ArraySetAsSeries(percentile_4h_100_50, true);
   percentile_4h_100_80_handle=iCustom(dataSymbol, tf_middle, "bn_percentile", 30, 80);
   ArraySetAsSeries(percentile_4h_100_80, true);
   percentile_4h_100_90_handle=iCustom(dataSymbol, tf_middle, "bn_percentile", 30, 90);
   ArraySetAsSeries(percentile_4h_100_90, true);   
   
   //1 DAY
   rsi_1d_handle = iCustom(dataSymbol,tf_large,"bn_RSI",14);
   ArraySetAsSeries(rsi_1d, true);
   
   ArrayInitialize(ma_1d_handles, -1);
   ma_1d_periods[0]=1;  ma_1d_periods[1]=10;    ma_1d_periods[2]=20;    ma_1d_periods[3]=30;
   ma_1d_periods[4]=50; ma_1d_periods[5]=100;
   
   ma_1d_count = ArraySize(ma_1d_handles);

   for (int ma=0; ma<ma_1d_count; ma++)
   {
      ma_1d_handles[ma]=iMA(dataSymbol, tf_large, ma_1d_periods[ma], 0, MODE_EMA, PRICE_CLOSE);
      ArraySetAsSeries(ma_1d_matrix[ma].line, true);
   }
   
   stoch_1d_handle = iStochastic(dataSymbol,tf_large, 5, 3, 3,MODE_SMA,STO_LOWHIGH);
   ArraySetAsSeries(stoch_1d[0].line, true);
   ArraySetAsSeries(stoch_1d[1].line, true);
            
   macd_1d_handle = iMACD(dataSymbol, tf_large, 12, 26, 9, PRICE_CLOSE);
   ArraySetAsSeries(macd_1d[0].line, true);
   ArraySetAsSeries(macd_1d[1].line, true);
   
   demarker_1h_handle = iDeMarker(dataSymbol,tf_small,14);
   ArraySetAsSeries(demarker_1h, true);
   wpr_1h_handle = iWPR(dataSymbol,tf_small,14);
   ArraySetAsSeries(wpr_1h, true);
   momentum_1h_handle = iMomentum(dataSymbol,tf_small,14, PRICE_CLOSE);
   ArraySetAsSeries(momentum_1h, true);
   
   /*
   demarker_1h_handle = iDeMarker(dataSymbol,tf_small,14);
   ArraySetAsSeries(demarker_1h, true);
   wpr_1h_handle = iWPR(dataSymbol,tf_small,14);
   ArraySetAsSeries(wpr_1h, true);
   momentum_1h_handle = iMomentum(dataSymbol,tf_small,14, PRICE_CLOSE);
   ArraySetAsSeries(momentum_1h, true);
   
   demarker_4h_handle = iDeMarker(dataSymbol,tf_middle,14);
   ArraySetAsSeries(demarker_4h, true);
   wpr_4h_handle = iWPR(dataSymbol,tf_middle,14);
   ArraySetAsSeries(wpr_4h, true);
   momentum_4h_handle = iMomentum(dataSymbol,tf_middle,14, PRICE_CLOSE);
   ArraySetAsSeries(momentum_4h, true);
   
   demarker_1d_handle = iDeMarker(dataSymbol,tf_large,14);
   ArraySetAsSeries(demarker_1d, true);
   wpr_1d_handle = iWPR(dataSymbol,tf_large,14);
   ArraySetAsSeries(wpr_1d, true);
   momentum_1d_handle = iMomentum(dataSymbol,tf_large,14, PRICE_CLOSE);
   ArraySetAsSeries(momentum_1d, true);
   */
   
   ArraySetAsSeries(time_1h, true);
   ArraySetAsSeries(time_4h, true);
   ArraySetAsSeries(time_1d, true);
   ArraySetAsSeries(open_1h, true);
   ArraySetAsSeries(open_4h, true);
   ArraySetAsSeries(open_1d, true);
   ArraySetAsSeries(high_1h, true);
   ArraySetAsSeries(high_4h, true);
   ArraySetAsSeries(high_1d, true);
   ArraySetAsSeries(low_1h, true);
   ArraySetAsSeries(low_4h, true);
   ArraySetAsSeries(low_1d, true);
   ArraySetAsSeries(close_1h, true);
   ArraySetAsSeries(close_4h, true);
   ArraySetAsSeries(close_1d, true);
   
}
void mt_ai_data_update()
{
   int bars_h1=Bars(dataSymbol, tf_small);
   CopyTime(dataSymbol, tf_small, 0, bars_h1, time_1h);
   //1 HOUR
   CopyOpen(dataSymbol, tf_small, 0, bars_h1, open_1h);
   CopyHigh(dataSymbol, tf_small, 0, bars_h1, high_1h);
   CopyLow(dataSymbol, tf_small, 0, bars_h1, low_1h);
   CopyClose(dataSymbol, tf_small, 0, bars_h1, close_1h);
   
   
   CopyBuffer(rsi_1h_handle,0,0,bars_h1,rsi_1h);
   
   for (int ma=0; ma<ma_1h_count; ma++)
   {
      CopyBuffer(ma_1h_handles[ma],0,0,bars_h1,ma_1h_matrix[ma].line);
   }                  
   
   CopyBuffer(stoch_1h_handle,0,0,bars_h1,stoch_1h[0].line);
   CopyBuffer(stoch_1h_handle,1,0,bars_h1,stoch_1h[1].line);
            
   CopyBuffer(bb_1h_handle, 0, 0, bars_h1, bb_1h[0].line);
   CopyBuffer(bb_1h_handle, 1, 0, bars_h1, bb_1h[1].line);
   
   CopyBuffer(macd_1h_handle, 0, 0, bars_h1, macd_1h[0].line);
   CopyBuffer(macd_1h_handle, 1, 0, bars_h1, macd_1h[1].line);
   
   CopyBuffer(demarker_1h_handle,0,0,bars_h1,demarker_1h);
   CopyBuffer(wpr_1h_handle, 0, 0, bars_h1, wpr_1h);
   CopyBuffer(momentum_1h_handle, 0, 0, bars_h1, momentum_1h);
   
   CopyBuffer(aroon_1h_handle, 0, 0, bars_h1, aroon_1h[0].line);
   CopyBuffer(aroon_1h_handle, 1, 0, bars_h1, aroon_1h[1].line);
   
   CopyBuffer(lrs_1h_handle,0,0,bars_h1,lrs_1h);
   CopyBuffer(r2_1h_handle, 0, 0, bars_h1, r2_1h);
   CopyBuffer(rmo_1h_handle, 0, 0, bars_h1, rmo_1h);
   
   
   CopyBuffer(percentile_1h_100_50_handle, 0, 0, bars_h1, percentile_1h_100_50);
   CopyBuffer(percentile_1h_100_80_handle, 0, 0, bars_h1, percentile_1h_100_80);
   CopyBuffer(percentile_1h_100_90_handle, 0, 0, bars_h1, percentile_1h_100_90);
   
   //4 HOUR
   int bars_h4=Bars(dataSymbol, tf_middle);
   CopyTime(dataSymbol, tf_middle, 0, bars_h4, time_4h);
   CopyOpen(dataSymbol, tf_middle, 0, bars_h4, open_4h);
   CopyHigh(dataSymbol, tf_middle, 0, bars_h4, high_4h);
   CopyLow(dataSymbol, tf_middle, 0, bars_h4, low_4h);
   CopyClose(dataSymbol, tf_middle, 0, bars_h4, close_4h);
   
   CopyBuffer(rsi_4h_handle,0,0,bars_h4,rsi_4h);
   
   for (int ma=0; ma<ma_4h_count; ma++)
   {
      CopyBuffer(ma_4h_handles[ma],0,0,bars_h4,ma_4h_matrix[ma].line);
   }                  
   
   CopyBuffer(bb_4h_handle, 0, 0, bars_h4, bb_4h[0].line);
   CopyBuffer(bb_4h_handle, 1, 0, bars_h4, bb_4h[1].line);
   CopyBuffer(bb_4h_handle, 2, 0, bars_h4, bb_4h[2].line);
   
   CopyBuffer(stoch_4h_handle,0,0,bars_h4,stoch_4h[0].line);
   CopyBuffer(stoch_4h_handle,1,0,bars_h4,stoch_4h[1].line);
            
   CopyBuffer(macd_4h_handle, 0, 0, bars_h4, macd_4h[0].line);
   CopyBuffer(macd_4h_handle, 1, 0, bars_h4, macd_4h[1].line);
   
   CopyBuffer(demarker_4h_handle,0,0,bars_h4,demarker_4h);
   CopyBuffer(wpr_4h_handle, 0, 0, bars_h4, wpr_4h);
   CopyBuffer(momentum_4h_handle, 0, 0, bars_h4, momentum_4h);
   
   
   CopyBuffer(percentile_4h_100_50_handle, 0, 0, bars_h4, percentile_4h_100_50);
   CopyBuffer(percentile_4h_100_80_handle, 0, 0, bars_h4, percentile_4h_100_80);
   CopyBuffer(percentile_4h_100_90_handle, 0, 0, bars_h4, percentile_4h_100_90);
   
            
   //1 DAY
   int bars_d1=Bars(dataSymbol, tf_large);
   CopyTime(dataSymbol, tf_large, 0, bars_d1, time_1d);
   CopyOpen(dataSymbol, tf_large, 0, bars_d1, open_1d);
   CopyHigh(dataSymbol, tf_large, 0, bars_d1, high_1d);
   CopyLow(dataSymbol, tf_large, 0, bars_d1, low_1d);
   CopyClose(dataSymbol, tf_large, 0, bars_d1, close_1d);
   
   CopyBuffer(rsi_1d_handle,0,0,bars_d1,rsi_1d);
   
   for (int ma=0; ma<ma_1d_count; ma++)
   {
      CopyBuffer(ma_1d_handles[ma],0,0,bars_d1,ma_1d_matrix[ma].line);
   }                  
   
   CopyBuffer(stoch_1d_handle, 0, 0, bars_d1, stoch_1d[0].line);
   CopyBuffer(stoch_1d_handle, 1, 0, bars_d1, stoch_1d[1].line);
            
   CopyBuffer(macd_1d_handle, 0, 0, bars_d1, macd_1d[0].line);
   CopyBuffer(macd_1d_handle, 1, 0, bars_d1, macd_1d[1].line);
   
   CopyBuffer(demarker_1d_handle,0,0,bars_d1,demarker_1d);
   CopyBuffer(wpr_1d_handle, 0, 0, bars_d1, wpr_1d);
   CopyBuffer(momentum_1d_handle, 0, 0, bars_d1, momentum_1d);
   
   
}


      
string mt_ai_data_get(int bar)
{
   Print(TimeToString(time_1h[bar]));
   MqlDateTime bar_time;
   TimeToStruct(time_1h[bar], bar_time);
   int bar_h4=iBarShift(dataSymbol,tf_middle,time_1h[bar],false);
   int bar_d1=iBarShift(dataSymbol,tf_large,time_1h[bar],false);
   
   string res=TimeToString(time_1h[bar])+";"+IntegerToString(bar_time.day_of_week)+";"+IntegerToString(bar_time.hour)+";";
   res+=DoubleToString(close_1h[bar],4)+";"
      +DoubleToString(open_1h[bar],4)+";"
      +DoubleToString((open_1h[bar]-low_1h[bar])/open_1h[bar],4)+";"
      +DoubleToString((high_1h[bar]-open_1h[bar])/open_1h[bar],4)+";"
      +DoubleToString(close_1h[bar]/open_1h[bar]-1,4)+";"
      +DoubleToString(close_1h[bar]/open_1h[bar]-1,4)+";";
   
   for (int b=1;b<=days_in_vector_1;b++)
   {
      res+=DoubleToString(close_4h[bar_h4+b]/open_4h[bar_h4+b]-1, 2)+";";
      double candle_size_buf = 0.00001 + high_4h[bar_h4+b]-low_4h[bar_h4+b];
      res+=DoubleToString((high_4h[bar_h4+b]-MathMax(open_4h[bar_h4+b], close_4h[bar_h4+b]))/candle_size_buf,2)+";";
      res+=DoubleToString((MathMin(open_4h[bar_h4+b], close_4h[bar_h4+b])-low_4h[bar_h4+b])/candle_size_buf,2)+";";
      //res+=DoubleToString(MathAbs(open_4h[bar+b] - close_4h[bar+b])/candle_size_buf,2)+";";
      
      res+=DoubleToString((high_4h[bar_h4+b]-high_4h[bar_h4+b+1])/high_4h[bar_h4+b+1],2)+";";
      res+=DoubleToString((low_4h[bar_h4+b]-low_4h[bar_h4+b+1])/low_4h[bar_h4+b+1],2)+";";
      
      //RSI  просто значение если нужны будут коридоры то 30 70 для часа
      res += DoubleToString(rsi_4h[bar_h4+b],2)+";";
      res += DoubleToString(1-rsi_4h[bar_h4+b]/rsi_4h[bar_h4+b+1],2)+";";
      res += DoubleToString(rsi_4h[bar_h4+b]-rsi_4h[bar_h4+b+1],2)+";";
      if (rsi_4h[bar_h4+b]>70) res += "1;0;";
      else if (rsi_4h[bar_h4+b]<30) res += "0;1;";
      else res+="0;0;";
      
      
      double buf = MathAbs(bb_4h[1].line[bar_h4+b]-bb_4h[0].line[bar_h4+b]);
      double deviation = (close_4h[bar_h4+b]-bb_4h[0].line[bar_h4+b])/buf;
      res += DoubleToString(deviation,2)+";";
      if (deviation>3) res+="1;0;0;";
      else if (deviation>2) res+="0;1;0;";
      else if (deviation>1) res+="0;0;1;";
      else res+="0;0;0;";
      if (deviation<-3) res+="1;0;0;";
      else if (deviation<-2) res+="0;1;0;";
      else if (deviation<-1) res+="0;0;1;";
      else res+="0;0;0;";
      
   }
   
   for (int b=1;b<=days_in_vector_2;b++)
   {
      res+=DoubleToString(close_1h[bar+b]/open_1h[bar+b]-1, 2)+";";
      double candle_size_buf = 0.00001 + high_1h[bar+b]-low_1h[bar+b];
      res+=DoubleToString((high_1h[bar+b]-MathMax(open_1h[bar+b], close_1h[bar+b]))/candle_size_buf,2)+";";
      res+=DoubleToString((MathMin(open_1h[bar+b], close_1h[bar+b])-low_1h[bar+b])/candle_size_buf,2)+";";
      //res+=DoubleToString(MathAbs(open_1h[bar+b] - close_1h[bar+b])/candle_size_buf,2)+";";
      
      res+=DoubleToString((high_1h[bar+b]-high_1h[bar+b+1])/high_1h[bar+b+1],2)+";";
      res+=DoubleToString((low_1h[bar+b]-low_1h[bar+b+1])/low_1h[bar+b+1],2)+";";
      
      //RSI  просто значение если нужны будут коридоры то 30 70 для часа
      res += DoubleToString(rsi_1h[bar+b],2)+";";
      res += DoubleToString(1-rsi_1h[bar+b]/rsi_1h[bar+b+1],2)+";";
      res += DoubleToString(rsi_1h[bar+b]-rsi_1h[bar+b+1],2)+";";
      if (rsi_1h[bar+b]>70) res += "1;0;";
      else if (rsi_1h[bar+b]<30) res += "0;1;";
      else res+="0;0;";
      
      
      double buf = MathAbs(bb_1h[1].line[bar+b]-bb_1h[0].line[bar+b]);
      double deviation = (ma_1h_matrix[MA_1].line[bar+b]-bb_1h[0].line[bar+b])/buf;
      res += DoubleToString(deviation,2)+";";
      if (deviation>3) res+="1;0;0;";
      else if (deviation>2) res+="0;1;0;";
      else if (deviation>1) res+="0;0;1;";
      else res+="0;0;0;";
      if (deviation<-3) res+="1;0;0;";
      else if (deviation<-2) res+="0;1;0;";
      else if (deviation<-1) res+="0;0;1;";
      else res+="0;0;0;";
      

   }
   
   
   return res;
}
void mt_ai_data_init_file()
{
   data_file_handle = FileOpen("data_for_ai_"+dataSymbol+".csv",FILE_ANSI|FILE_COMMON|FILE_WRITE,";");
   string heading="datetime;day;hour;close_current;open_current;move_down;move_up;percent;percent_10";
   int indicators_count=17;
   for (int i=1;i<=(days_in_vector_1+days_in_vector_2)*indicators_count;i++)
      heading+=";in_"+IntegerToString(i);//in_1;in_2;in_3;in_4;in_5;in_6;in_7;in_8;in_9;in_10;in_11;in_12;in_13;in_14;in_15;in_16;in_17;in_18;in_019;in_20;in_21;in_22;in_23;in_24;in_25;in_26;in_27;in_28;in_29;in_30;in_31;in_32;in_33;in_34;in_35;in_36;in_37;in_38;in_39;in_40;in_41;in_42;in_43;in_44;in_45;in_46;in_47";
   FileWrite(data_file_handle, heading);
}      
void mt_ai_data_save_file(string line_to_save)
{
   //Print(line_to_save);
   FileWrite(data_file_handle, line_to_save);
}
void mt_ai_data_close_file()
{
   FileClose(data_file_handle);
}    