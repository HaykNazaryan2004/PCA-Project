# Երկու ձևով տվյալների վրա կիռարել ենք ներոնային ցանց, որպեսզի հասկանանք PCA-ի ազդեցությունը ուսուցման ժամանակի և ճշտության վրա։ 
# Առաջին կոդում նվազեցրել ենք չափողականությունը  PCA-ի միջոցով հետո կիրառել նեյրոնային ցանց։ Իսկ առաջին կոդում միանգամից կիրառել ենք նեյրոնային ցանց։
# Սիդը նվիրված է ԵՊՀ-ի 106 ամյակին



# Նեռբեռնել ենք անրաժեշտ գրադարանները 
install.packages("keras")
install.packages("tensorflow")
install.packages("caret")
install.packages("dplyr")


library(keras)
library(tensorflow)
library(caret)
library(dplyr)

# Ֆիքսենք սիդ
set.seed(106)

# Fashion mnist տվյալների ներբեռնում
fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Նկարները դարձնենք միաչափ վեկտորներ (28x28 դարձնենք 784x1)
train_images <- array_reshape(train_images, c(nrow(train_images), 784)) / 255
test_images <- array_reshape(test_images, c(nrow(test_images), 784)) / 255

#  PCA-ի կիրառում
pca_model <- prcomp(train_images, center = TRUE, scale. = TRUE)

# Գտնենք այն կոպոնենտների քանակը(գծային տարածության չափողականությունը) որոնք պահանում են 95% վարիացիա
explained_variance <- cumsum(pca_model$sdev^2 / sum(pca_model$sdev^2))
num_components <- which(explained_variance >= 0.95)[1]
cat("կոմպոնենտների քանակը, որոնք պարունակում են  95% վարիացիա:", num_components, "\n")

# բաժանենք ուսուցանվող և թեստավորվող տվյաներ(train-ի և test-ի բաժանում)
train_pca <- predict(pca_model, train_images)[, 1:num_components]
test_pca <- predict(pca_model, test_images)[, 1:num_components]

# կատերգորիկ փոփոխականները դարձրել ենք թվային
train_labels_cat <- to_categorical(train_labels, 10)
test_labels_cat <- to_categorical(test_labels, 10)

# կառուցենք նեյրոնային ցանց
model_raw <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = 'accuracy'
)

# չափենք ուսուցման ժամանակը
start_time_pca <- Sys.time()

# ուսուցանել մոդելը
history <- model %>% fit(
  x = train_pca,
  y = train_labels_cat,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2,
  verbose = 2

  cat("Ուսուցման տևողությունը PCA-ով:", training_time_pca, "\n")
)

# գնահատենք ճշտությունը տեստ(test)- դատայի վրա
score <- model %>% evaluate(test_pca, test_labels_cat, verbose = 0)
cat("Թեստի ճշտությունը PCA-ով:", round(score[[2]] * 100, 2), "%\n")
