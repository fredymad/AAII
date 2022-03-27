# El conjunto de datos spam se puede encontrar
# http://archive.ics.uci.edu/ml/datasets/Spambase
# http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/
# https://hastie.su.domains/ElemStatLearn/

# Bibliografia
# https://rpubs.com/phamdinhkhanh/389752
# https://rstudio-pubs-static.s3.amazonaws.com/291503_305320e8ca5d48a9928989a63b92789b.html
# https://towardsdatascience.com/what-is-out-of-bag-oob-score-in-random-forest-a7fa23d710
# https://rpubs.com/archita25/677243
# https://bradleyboehmke.github.io/HOML/random-forest.html
# https://link.springer.com/article/10.1023/A:1010933404324
# https://wires.onlinelibrary.wiley.com/doi/10.1002/widm.1301
# https://stackoverflow.com/questions/22909197/creating-folds-for-k-fold-cv-in-r-using-caret

# Borramos cualquier ejecución previa
rm(list = ls()) 

# Fijamos el directorio de trabajo
setwd("~/Desktop/Máster/Segundo cuatrimestre/Obligatorias/AAII/PEC 1")

# Fijamos semilla
set.seed(1)

# Cargamos la libreria de randomForest
library(randomForest)
library(caret)
library(plotly)
library(tidyverse)

# Cargamos los datos y los juntamos en un único conjunto
#spam <- cbind(read.table(file = "spam.traintest.txt", col.names = c("set")),
#              read.table(file = "spam.data.txt"))

spam <- read.table(file = "spam.data.txt")

# Transformamos la variable respuesta en factor
spam[,ncol(spam)] <- as.factor(spam[,ncol(spam)])

# Vemos el conjunto de datos
Hmisc::describe(spam)
str(spam)

# Creamos los folds de forma estratificada
folds <- createFolds(spam$V58)
split_up <- lapply(folds, function(ind, dat) dat[ind,], dat = spam)
names(split_up) <- c("s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10")
list2env(split_up, envir = .GlobalEnv)

# Implementamos CV manual
for(i in 1:10){
   if(i == 1){
     entrenamiento <- rbind(s1, s2, s3, s4, s5, s6, s7, s8, s9)
     validacion <- s10
   }
   else if (i == 2){
     entrenamiento <- rbind(s2, s3, s4, s5, s6, s7, s8, s9, s10)
     validacion <- s1
   }
   else if (i == 3){
     entrenamiento <- rbind(s3, s4, s5, s6, s7, s8, s9, s10, s1)
     validacion <- s2
   }
   else if (i == 4){
     entrenamiento <- rbind(s4, s5, s6, s7, s8, s9, s10, s1, s2)
     validacion <- s3
   }
   else if (i == 5){
     entrenamiento <- rbind(s5, s6, s7, s8, s9, s10, s1, s2, s3)
     validacion <- s4
   }
   else if (i == 6){
     entrenamiento <- rbind(s6, s7, s8, s9, s10, s1, s2, s3, s4)
     validacion <- s5
   }
   else if (i == 7){
     entrenamiento <- rbind(s7, s8, s9, s10, s1, s2, s3, s4, s5)
     validacion <- s6
   }
   else if (i == 8){
     entrenamiento <- rbind(s8, s9, s10, s1, s2, s3, s4, s5, s6)
     validacion <- s7
   }
   else if (i == 9){
     entrenamiento <- rbind(s9, s10, s1, s2, s3, s4, s5, s6, s7)
     validacion <- s8
   }
   else if (i == 10){
     entrenamiento <- rbind(s10, s1, s2, s3, s4, s5, s6, s7, s8)
     validacion <- s9
   }
   print(paste0("k-fold:", i))

# Cambiamos el número de bosques, probamos con cuatro valores
for(ntree_parameter in c(100, 250, 500, 1000)){
  # Entrenamos con número variable de mtry desde 1 hasta 57 (máximo número de columnas)
  for (mtry_parameter in seq(1:(ncol(entrenamiento)-1))){
    # Hora de inicio de la ejecución
    start.time <- Sys.time()
    # Entrenamiento del algoritmo
    rf <- randomForest(as.formula(paste(names(entrenamiento)[ncol(entrenamiento)],  "~ .")), data=entrenamiento, ntree=ntree_parameter, mtry = mtry_parameter)
    # Fin de la ejecución
    end.time <- Sys.time()
    # Hacemos las predicciones sobre el conjunto de validación
    prediccion <- predict(rf, validacion, type = "response")
    # Obtenemos la matriz de confusión
    confusion <- confusionMatrix(factor(prediccion), factor(validacion[,ncol(validacion)]), positive = "1")
    # Promediamos el error de los árboles para saber cual es el error oob
    oob_eror <- mean(rf$err.rate[,1])
    # Calculamos el error de test
    cv_error <- 1 - confusion$overall[["Accuracy"]]
    # Calculamos el tiempo total de ejecución
    execution_time <- end.time-start.time
    # Guardamos en un txt los resultados
    resultado = data.frame("oob_error" = oob_eror, "oob_error_sd" = oob_error_sd, "cv_error" = cv_error, "ntree" = ntree_parameter, "mtry" = mtry_parameter, "execution_time" = execution_time, "iteracion" = i)
    #write.table(resultado, file ="resultadoNuevov6.txt", col.names = FALSE, append = TRUE, row.names = F)
  }
}
}

# Cargamos el archivo guardado
resultado <- read.table(file = "resultadoNuevov5.txt", col.names = c("oob_error", "cv_error", "ntree", "mtry", "execution_time", "iteracion"))

# Promediamos el error
resultado <- resultado %>%
  group_by(ntree, mtry) %>%
  summarise_at(vars(oob_error, cv_error, execution_time), funs(mean, sd))

# Nos quedamos únicamente con un experimento
resultado <- resultado %>%
  #filter(ntree == 100) 
  filter(ntree == 250) 
  #filter(ntree == 500) 
  #filter(ntree == 1000)

resultado <- resultado %>%
  mutate(cv_max = cv_error_mean + cv_error_sd,
         cv_min = cv_error_mean - cv_error_sd,
         oob_max = oob_error_mean + oob_error_sd,
         oob_min = oob_error_mean - oob_error_sd)

# Hacemos la representación

# Prueba
fig <- plot_ly(resultado, x = ~mtry, y = ~oob_error_mean, name = 'OOB error', type = 'scatter', mode = 'lines',
               line = list(color = 'rgb(255, 0, 0)', width = 3)) 

fig <- fig %>%
  add_ribbons(ymin=~oob_min,
              ymax=~oob_max,
              line=list(color='rgb(51, 53, 255)', width=0),
              fillcolor = 'rgba(255, 153, 153, 0.4)',
              showlegend = FALSE)
fig

fig <- fig %>%
  add_trace(y = ~cv_error_mean, name = 'CV error', line = list(color = 'rgb(51, 53, 255)', width = 3)) %>%
  add_ribbons(ymin=~cv_min,
              ymax=~cv_max,
              line=list(color='rgb(51, 53, 255)', width=0),
              fillcolor = 'rgba(7, 164, 181, 0.2)',
              showlegend = FALSE)

fig

fig <- fig %>% add_trace(
  x = ~mtry,
  y = ~execution_time_mean,
  yaxis = "y2",
  name = 'Time',
  line = list(color = 'rgba(7, 164, 181, 100)', width = 3))

fig <- fig %>% layout(title = "CV and OBB error (250 trees)",
                      xaxis = list(tickvals = seq(0, 60, by = 5), 
                                   title = "mtry (number of predictors)"),
                      yaxis = list(tickvals = seq(0, 0.1, by = 0.005),
                                   range = c(0, 0.1),
                                   title = "Classification error (%)", tickformat = ".1%"),
                      yaxis2 = list(
                        #tickfont = list(color = "#d62728"),
                        #titlefont = list(color = "#d62728"),
                        overlaying = "y",
                        side = "right",
                        title = "Execution time (s)"))
fig












 
# Tiempo de ejecución
# Elegir en base al número de árboles
# Cargamos el archivo guardado
resultado <- read.table(file = "resultadoNuevov5.txt", col.names = c("OOB_error", "CV_error", "ntree", "mtry", "execution_time", "iteracion"))
resultado <- resultado %>% group_by(ntree, mtry) %>% summarise_all(mean) %>% select(-iteracion)
resultado <- resultado %>% select(-execution_time) %>% pivot_wider(names_from = ntree, values_from = c("OOB_error", "CV_error"))

fig <- plot_ly(resultado, x = ~mtry, y = ~OOB_error_100, name = 'OOB error 100', type = 'scatter', mode = 'lines',
               line = list(color = 'rgb(255, 0, 0)', width = 3)) 
fig <- fig %>% add_trace(y = ~OOB_error_250, name = 'OOB error 250', line = list(color = 'rgb(255, 51, 51)', width = 3))
fig <- fig %>% add_trace(y = ~OOB_error_500, name = 'OOB error 500', line = list(color = 'rgb(255, 102, 102)', width = 3))
fig <- fig %>% add_trace(y = ~OOB_error_1000, name = 'OOB error 1000', line = list(color = 'rgb(255, 153, 153)', width = 3))

fig <- fig %>% add_trace(y = ~CV_error_100, name = 'CV error 100', line = list(color = 'rgb(51, 53, 255)', width = 3))
fig <- fig %>% add_trace(y = ~CV_error_250, name = 'CV error 250', line = list(color = 'rgb(0, 128, 255)', width = 3))
fig <- fig %>% add_trace(y = ~CV_error_500, name = 'CV error 500', line = list(color = 'rgb(102, 178, 255)', width = 3))
fig <- fig %>% add_trace(y = ~CV_error_1000, name = 'CV error 1000', line = list(color = 'rgb(153, 204, 255)', width = 3))
fig <- fig %>% layout(title = "CV and OOB errors",
                      xaxis = list(tickvals = seq(0, 60, by = 5), 
                                   title = "mtry (number of predictors)"),
                      yaxis = list(tickvals = seq(0, 0.1, by = 0.005),
                                   range = c(0, 0.1),
                                   title = "Classification error (%)", tickformat = ".1%"))
fig

fig <- fig %>% layout(title = "CV and OOB errors",
                      xaxis = list(tickvals = seq(0, 60, by = 5), title = "mtry (number of predictors)"),
                      yaxis = list(tickvals = seq(0, 0.1, by = 0.005), range = c(0, 0.1), title = "Classification error (%)", tickformat = ".1%"))
fig

