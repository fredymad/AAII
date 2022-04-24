# Cargamos las librerias de interés
# Imputación de datos
# library(mice)
# library(DMwR)
library(Hmisc)
library(dplyr)
library(VIM)
library(factoextra)
library(NbClust)
library(cluster)
library(readxl)
library(tidyverse)
library(mclust)
library(plotly)
library(corrplot)
library(scatterplot3d)
library(clusterSim)
library(ClusterR)

# https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/#fviz_nbclust-function-elbow-silhouhette-and-gap-statistic-methods
# Fijamos el directorio de trabajo
setwd("~/Desktop")

# Cargamos los datos
data <- read_csv("data.csv", col_names = F)

# Representamos los datos
plot(data)

# Aseguramos replicabilidad en los experimentos
set.seed(1)

# Número de valores perdidos
sum(is.na(data))

# Análisis exploratorio
summary(data)

# Histograma
hist.data.frame(data)

# Rango de valores
data %>% summarise_all(max, na.rm = T) - data %>% summarise_all(min, na.rm = T)
#data %>% summarise_all(sd, na.rm = T)
#data %>% summarise_all(funs(max, min))

# Contamos los outliers
sd_mean <- function(x){
  length(which(x > mean(x) + 3 * sd(x) | x < mean(x) - 3 * sd(x)))
}

leve <- function(x){
  length(which(x > quantile(x)[4] + 1.5*IQR(x) | x < quantile(x)[2] - 1.5 * IQR(x)))
}

extremo <- function(x){
  length(which(x > quantile(x)[4] + 3*IQR(x) | x < quantile(x)[2] - 3 * IQR(x)))
}

x <- sapply(data, extremo)
data %>%  summarise_all(funs(sd_mean, leve, extremo))

# Estandarizamos los datos
data <- data %>%
  mutate_each(funs(scale))

# Dos formas de procesar los datos, imputando o eliminando los valores perdidos
# A) Imputación de valores perdidos
data <- kNN(data, k = 3)
# Nos quedamos con las variables imputadas
data <- data[,c(1:10)]

# B) Eliminación de los valores perdidos
#data <- data[complete.cases(data),]

# Representamos los histogramas antes de normalizar
hist.data.frame(data)

# Calculamos la correlación de los datos
correlacion <- data %>%
  select_if(is.numeric) %>%
  cor(method = c("pearson")) %>%
  round(digits = 2)

# Correlación de Pearson
corrplot(correlacion, method = 'number', type = "upper")

# Visualización de los datos mediante violín
vis <- data %>%
  pivot_longer(cols = everything()) %>%
  mutate()

plot_ly(
  data = vis,
  x = ~name,
  y = ~value,
  type = "violin",
  color = ~name,
  #side = "positive",
  meanline = list(visible = T)
) %>%
  layout(title = "Dispersión",
         xaxis = list(title = "Variables",
                     categoryorder = "array",
                      categoryarray = c("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10")),
         yaxis = list(title = "Valores"))

# Visualización de boxplot
plot_ly(
  data = vis,
  x = ~name,
  y = ~value,
  type = "box",
  color = ~name,
  #side = "positive",
  meanline = list(visible = T)
)

# Guardamos el conjunto de datos procesado
# saveRDS(data, file = "datosProcesados.RDS")


####### K-MEANS #######

# Cargamos el conjunto de datos procesado
#data <- readRDS("~/Desktop/datosProcesados.RDS")
data <- readRDS("~/Desktop/datosProcesadosCompleteCases.RDS")

# Si queremos hacer una ejecución
#lista_kmeans = list()
# Seleccionamos el número de clústeres
#for (i in c(1:10)){
#  lista_kmeans[i] <- kmeans(data, i, nstart = 30)}
# Ejecutamos k-means con ambos números de clústeres
#result <- kmeans(data, 3, nstart = 30)
#result

# Elbow method
kmeans_wss <- fviz_nbclust(data, kmeans, nstart = 30, k.max = 10, method = "wss")

# Representación gráfica
kmeans_wss +
  geom_vline(xintercept = 5, linetype = 2) +
  labs(subtitle = "Elbow method")
round(kmeans_wss[["data"]]$y, digits = 3)

# Silhouette method
kmeans_siluetas <- fviz_nbclust(data, kmeans, nstart = 30, k.max = 10, method = "silhouette")
kmeans_siluetas +
  labs(subtitle = "Silhouette method")

# Gap
# Problema de convergencia, hay que aumentar el número de iteraciones
kmeans_Gap <- fviz_nbclust(data, kmeans, nstart = 30, iter.max = 30, k.max = 10, method = "gap_stat", nboot = 500)
kmeans_Gap + labs(subtitle = "Gap statistic method")

# Cálculo de métricas de validación internas iteranco
# Creamos vectores donde almacenar el valor de AIC y de BIC global
DB <- vector()
CH <- vector()

for (i in c(1:10)) {
  print(i)
  ejecucion <- kmeans(data, i, nstart = 30)
  db <- index.DB(data, ejecucion$cluster, centrotypes="centroids")
  ch <- index.G1(data, ejecucion$cluster, centrotypes="centroids")
  DB <- c(DB, db[["DB"]])
  CH <- c(CH, ch)
  }

indices_manuales <- cbind(kmeans_wss[["data"]], kmeans_siluetas[["data"]][["y"]], kmeans_Gap[["data"]][["gap"]], CH, DB)
indices_manuales$clusters <- as.numeric(as.character(indices_manuales$clusters))
indices_manuales <- round(indices_manuales, digits = 3)
write.table(indices_manuales, file = "indices_manuales.txt", row.names = F)

# Cálculo de 30 índices con paquete NbClust
indices <- NbClust(data = data, distance = "euclidean",
       # min.nc = 2, max.nc = 10, method = "kmeans", index = "alllong")
       min.nc = 2, max.nc = 10, method = "kmeans", index = "all")

# Vemos los mejores métodos y los resultados
prueba <- round(as.data.frame(t(indices[["Best.nc"]])) %>% arrange(Number_clusters), digits = 2)
write.table(prueba, file = "indices.txt", row.names = T, sep = "\t")

# Sólo nos quedamos con las métricas que eligen k clústers
prueba <- prueba %>%
  mutate(nombres = row.names(prueba)) %>%
  dplyr::select(Number_clusters, nombres) %>%
  group_by(Number_clusters) %>%
  mutate(d = paste0(nombres, collapse = " ")) %>%
  dplyr::select(Number_clusters, d) %>%
  mutate(n = max(row_number())) %>%
  distinct()
write.table(prueba, file = "indices2.txt", row.names = T, sep = "\t")

# Visualización de los clústeres
for (i in c(2:10)) {
  print(i)
  res.km <- kmeans(data, i, nstart = 30)
  orca(fviz_cluster(res.km, data = data,
                    #palette = c("#2E9FDF", "#00AFBB", "#E7B800"), 
                    geom = "point",
                    ellipse.type = "convex", 
                    ggtheme = theme_bw()), file = paste0("km", i, ".png"))
  #Obtenemos la figura de siluetas
  tryCatch({
    # Cuando k= 1 da error
    # Para imprimir las imágenes descomentar la línea de abajo
    # orca(fviz_silhouette(silhouette(res.km$cluster, dist(data))), file = paste0("km_sil", i, ".png"))
  }, error=function(e){})
  }

############## GMM ##############
#https://cran.r-project.org/web/packages/ClusterR/vignettes/the_clusterR_package.html

#data <- readRDS("~/Desktop/datosProcesados.RDS")
data <- readRDS("~/Desktop/datosProcesadosCompleteCases.RDS")

# Creamos vectores donde almacenar el valor de AIC y de BIC global
AIC_values <- vector()
BIC_values <- vector()
DB <- vector()
CH <- vector()
S <- vector()

for (i in c(2:10)){
  mc <- Mclust(data, G = i)
  # Indicamos el estado de la ejecución
  print(i)
  # Debemos tener en cuenta, que Mclust calcula múltiples valores
  # en función de las constricciones: https://bradleyboehmke.github.io/HOML/model-clustering.html
  # Por lo tanto aquí estamos sacando el valor de AIC y de BIC global (es decir, el mejor de todos los distintos modelos que calcula)
  AIC_values <- c(AIC_values, AIC(mc))
  BIC_values <- c(BIC_values, BIC(mc))
  db <- index.DB(data, mc$classification)
  ch <- index.G1(data, mc$classification)
  s <- mean(silhouette(mc$classification, dist(data))[,3])
  DB <- c(DB, db[["DB"]])
  CH <- c(CH, ch)
  S <- c(S, s)
  
  # Representación gráfica de los clústeres
  tryCatch({
   # orca(fviz_mclust(mc, "classification", geom = "point",pointsize = 1.5, palette = "jco"), file = paste0("gmm", i, ".png"))
  }, error=function(e){})
  #x <- fviz_mclust(mc, "classification", geom = "point", pointsize = 1.5, palette = "jco")
}

indices_manuales <- cbind(AIC_values, BIC_values, S, CH, DB)
indices_manuales <- round(indices_manuales, digits = 3)
write.table(indices_manuales, file = "indices_manuales_gmm.txt", row.names = F)


# Para calcular el valor de BIC
mc <- Mclust(data, G = 2:10)
x <- as.data.frame(mc[["BIC"]][])
summary(mc)
plot(mc, what = 'BIC')

##
fviz_mclust_bic(
  mc,
  model.names = TRUE,
  shape = 19,
  color = "model",
  palette = NULL,
  legend = NULL,
  main = "Model selection",
  xlab = "Number of components",
  ylab = "BIC")

# Representación del mejor valor de AIC y de BIC (entre todos los modelos)
representacion <- as.data.frame(AIC_values)
representacion <- cbind(representacion,as.data.frame(BIC_values))
representacion$k <- as.numeric(row.names(representacion))
fig <- plot_ly(representacion, x = ~k,
               y = ~-AIC_values,
               name = "AIC", type = 'scatter', mode = 'lines')  %>%
  add_trace(x = ~k,
            y = ~-BIC_values,
            name = "BIC") %>%
  layout(xaxis=list(title ="k"),
         yaxis=list(title = "Information criterion", tickformat=',d')) %>%
  add_segments(x = 10, xend = 10, y = -9492, yend = -16000, line = list(color = '#1f77b4'), showlegend = FALSE) %>%
  add_segments(x = 8, xend = 8, y = -9492, yend = -16000,line = list(color = "#ff7f0e"), showlegend = FALSE)

fig

# Otra aproximación
# https://stats.stackexchange.com/questions/210439/which-metric-is-used-in-the-em-algorithm-for-gmm-training
gmm = GMM(data, 2, dist_mode = "maha_dist", seed_mode = "random_subset", km_iter = 10,
          em_iter = 10, verbose = F)
pr = predict(gmm, newdata = data)
opt_gmm = Optimal_Clusters_GMM(data, max_clusters = 10, criterion = "BIC", 
                               dist_mode = "maha_dist",
                               plot_data = T)



##### CLÚSTERING JERÁRQUICO #####
#https://www.r-bloggers.com/2017/12/how-to-perform-hierarchical-clustering-using-r/
# https://rpubs.com/jaimeisaacp/760355
# https://rpubs.com/mjimcua/clustering-jerarquico-en-r
# https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad#:~:text=To%20get%20the%20optimal%20number,the%20distance%20between%20those%20clusters.

# Cargamos los datos
#data <- readRDS("~/Desktop/datosProcesados.RDS")
data <- readRDS("~/Desktop/datosProcesadosCompleteCases.RDS")

# Calculamos la matriz de distancias
#d1 <- dist(data)
#res.hc <- hclust(d1, method = "complete")
#corte <- cutree(res.hc, k = 5)

# Índice CPCC
#d2 <- cophenetic(res.hc)
#cor(d1, d2) 

# Vemos los parámetros por defecto de la función hcut
# formals(hcut)
# Elbow method
hclust_wss <- fviz_nbclust(data, FUN = hcut, nstart = 30, k.max = 10, method = "wss")

# Representación gráfica
hclust_wss +
  geom_vline(xintercept = 5, linetype = 2) +
  labs(subtitle = "Elbow method")
round(kmeans_wss[["data"]]$y, digits = 3)

# Silhouette method
hclust_siluetas <- fviz_nbclust(data, FUN = hcut, nstart = 30, k.max = 10, method = "silhouette")
hclust_siluetas +
  labs(subtitle = "Silhouette method")

# Gap
# Problema de convergencia, hay que aumentar el número de iteraciones
hclust_Gap <- fviz_nbclust(data, FUN = hcut, nstart = 30, iter.max = 30, k.max = 10, method = "gap_stat", nboot = 500)
hclust_Gap + labs(subtitle = "Gap statistic method")

# Cálculo de métricas de validación internas iteranco
# Creamos vectores donde almacenar el valor de AIC y de BIC global
DB <- vector()
CH <- vector()

for (i in c(1:10)) {
  print(i)
  ejecucion <- hcut(data,
    k = i,
    hc_func = c("hclust"),
    hc_method = "ward.D2",
    hc_metric = "euclidean")
  
  db <- index.DB(data, ejecucion$cluster)
  ch <- index.G1(data, ejecucion$cluster)
  DB <- c(DB, db[["DB"]])
  CH <- c(CH, ch)
}

indices_manuales <- cbind(hclust_wss[["data"]], hclust_siluetas[["data"]][["y"]], hclust_Gap[["data"]][["gap"]], CH, DB)
indices_manuales$clusters <- as.numeric(as.character(indices_manuales$clusters))
indices_manuales <- round(indices_manuales, digits = 3)
write.table(indices_manuales, file = "indices_manuales_hclust.txt", row.names = F)

# Visualización de los clústeres
for (i in c(2:10)) {
  print(i)
  res.km <- hcut(data,
                 k = i,
                 hc_func = c("hclust"),
                 hc_method = "ward.D2",
                 hc_metric = "euclidean")
  orca(fviz_dend(res.km, show_labels = FALSE), file = paste0("hc", i, ".png"))
  #Obtenemos la figura de siluetas
  tryCatch({
    # Cuando k= 1 da error
    # Para imprimir las imágenes descomentar la línea de abajo
    # orca(fviz_silhouette(silhouette(res.km$cluster, dist(data))), file = paste0("km_sil", i, ".png"))
  }, error=function(e){})
}
