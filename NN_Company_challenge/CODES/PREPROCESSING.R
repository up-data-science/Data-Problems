#Modelos con imputación y preprocesamiento

library(Hmisc)
library(e1071)
library(caret)

train = dtrain[,-1]

#preprocessing the name of variables
s = colnames(train)
ss = str_replace_all(s,"_diff","dd")
ss = str_replace_all(ss, " ",".")
ss = str_replace_all(ss, "´(kmt)´","ktm")
ss = str_replace_all(ss, "\\s*\\([^\\)]+\\)","")
ss = str_sub(ss, 3, str_length(ss))

ss = ss[-c(1:2)] #with out mounth and country

sec = c(rep(1,10),rep(2,10),rep(3,10),rep(4,10),rep(5,10),rep(6,10),rep(7,10),rep(8,10)
        ,rep(9,10),rep(10,10),rep(11,10),rep(12,10))

sec = as.character(sec)
sec = c(sec,"1n")

sss = paste(ss, sec, sep="")
sss = c(s[c(1:2)],sss)
sss[123] = "Target"

colnames(train) = sss

train$Target = as.factor(train$Target)
levels(train$Target) <- c('down', 'up')

#number of missing values per column
sapply(train, function(x) sum(is.na(x)))

# calculate correlation matrix
correlationMatrix <- cor(dtrain[,3:123],use = "complete.obs") 
# summarize the correlation matrix
View(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.5)
# print indexes of highly correlated attributes
print(highlyCorrelated)

library(corrr)

cor.pairs = cor(train[,3:122],use = "complete.obs") %>%
  as.data.frame() %>%
  mutate(var1 = rownames(.)) %>%
  gather(var2, value, -var1) %>%
  arrange(desc(value)) %>%
  group_by(value) %>%
  filter(row_number()==1)
cor.pairs = cor.pairs[-1,]
corpairs1 = as.data.frame(cor.pairs)
setDT(melt(correlationMatrix))[order(value)]
#due to the autocorrelated behaviour of the data, we will try to use a way of imputation 

#we'll imput the data looking for the variables more correlated with the features that has missing values
#correlated features with the variables with more missing values

diffImports <- filter(corpairs1, grepl('dImports',var1))
dClosing.stocks <- filter(corpairs1, grepl('dClosing.stocks',var1))


#We will try to use the library Hmisc with the function aregImpute to find the best way of imputation to the two variables that have the most amount of missing values

train$dImports1 <- with(train, Hmisc::impute(dImports1, mean)); train$dImports1 = as.numeric(train$dImports1)
train$dImports2 <- with(train, Hmisc::impute(dImports2, mean)); train$dImports2 = as.numeric(train$dImports2)
train$dImports3 <- with(train, Hmisc::impute(dImports3, mean)); train$dImports3 = as.numeric(train$dImports3)
train$dImports4 <- with(train, Hmisc::impute(dImports4, mean)); train$dImports4 = as.numeric(train$dImports4)
train$dImports5 <- with(train, Hmisc::impute(dImports5, mean)); train$dImports5 = as.numeric(train$dImports5)
train$dImports6 <- with(train, Hmisc::impute(dImports6, mean)); train$dImports6 = as.numeric(train$dImports6)
train$dImports7 <- with(train, Hmisc::impute(dImports7, mean)); train$dImports7 = as.numeric(train$dImports7)
train$dImports8 <- with(train, Hmisc::impute(dImports8, mean)); train$dImports8 = as.numeric(train$dImports8)
train$dImports9 <- with(train, Hmisc::impute(dImports9, mean)); train$dImports9 = as.numeric(train$dImports9)
train$ddImports10 <- with(train, Hmisc::impute(ddImports10, mean)); train$ddImports10 = as.numeric(train$ddImports10)
train$ddImports11 <- with(train, Hmisc::impute(ddImports11, mean)); train$ddImports11 = as.numeric(train$ddImports11)
train$ddImports12 <- with(train, Hmisc::impute(ddImports12, mean)); train$ddImports12 = as.numeric(train$ddImports12)

#ddClosing.stocks
train$dClosing.stocks1 <- with(train, Hmisc::impute(dClosing.stocks1, mean)); train$dClosing.stocks1 = as.numeric(train$dClosing.stocks1)
train$dClosing.stocks2 <- with(train, Hmisc::impute(dClosing.stocks2, mean)); train$dClosing.stocks2 = as.numeric(train$dClosing.stocks2)
train$dClosing.stocks3 <- with(train, Hmisc::impute(dClosing.stocks3, mean)); train$dClosing.stocks3 = as.numeric(train$dClosing.stocks3)
train$dClosing.stocks4 <- with(train, Hmisc::impute(dClosing.stocks4, mean)); train$dClosing.stocks4 = as.numeric(train$dClosing.stocks4)
train$dClosing.stocks5 <- with(train, Hmisc::impute(dClosing.stocks5, mean)); train$dClosing.stocks5 = as.numeric(train$dClosing.stocks5)
train$dClosing.stocks6 <- with(train, Hmisc::impute(dClosing.stocks6, mean)); train$dClosing.stocks6 = as.numeric(train$dClosing.stocks6)
train$dClosing.stocks7 <- with(train, Hmisc::impute(dClosing.stocks7, mean)); train$dClosing.stocks7 = as.numeric(train$dClosing.stocks7)
train$dClosing.stocks8 <- with(train, Hmisc::impute(dClosing.stocks8, mean)); train$dClosing.stocks8 = as.numeric(train$dClosing.stocks8)
train$dClosing.stocks9 <- with(train, Hmisc::impute(dClosing.stocks9, mean)); train$dClosing.stocks9 = as.numeric(train$dClosing.stocks9)
train$ddClosing.stocks10 <- with(train, Hmisc::impute(ddClosing.stocks10, mean)); train$ddClosing.stocks10 = as.numeric(train$ddClosing.stocks10)
train$ddClosing.stocks11 <- with(train, Hmisc::impute(ddClosing.stocks11, mean)); train$ddClosing.stocks11 = as.numeric(train$ddClosing.stocks11)
train$ddClosing.stocks12 <- with(train, Hmisc::impute(ddClosing.stocks12, mean)); train$ddClosing.stocks12 = as.numeric(train$ddClosing.stocks12)

str(train)
