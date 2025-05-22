# գրադարանների ներբեռնում
library(keras)
library(tensorflow)

# սիդի ընտրություն
set.seed(106)

# Fashion mnist տվյալների ներբեռնում

fashion_mnist <- dataset_fashion_mnist()
c(train_images, train_labels) %<-% fashion_mnist$train
c(test_images, test_labels) %<-% fashion_mnist$test

# Նկարները դարձնենք միաչափ վեկտորներ (28x28 դարձնենք 784x1)
train_images <- array_reshape(train_images, c(nrow(train_images), 784)) / 255
test_images <- array_reshape(test_images, c(nrow(test_images), 784)) / 255

# կատերգորիկ փոփոխականները դարձրել ենք թվային
train_labels_cat <- to_categorical(train_labels, 10)
test_labels_cat <- to_categorical(test_labels, 10)

# # կառուցենք նեյրոնային ցանց
model_raw <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model_raw %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = 'accuracy'
)

# չափենք ուսուցման ժամանակը
start_time_pca <- Sys.time()

# ուսուցանել մոդելը
history_raw <- model_raw %>% fit(
  x = train_images,
  y = train_labels_cat,
  epochs = 20,
  batch_size = 128,
  validation_split = 0.2,
  verbose = 2
)

# գնահատենք ճշտությունը տեստ(test) դատայի վրա
score_raw <- model_raw %>% evaluate(test_images, test_labels_cat, verbose = 0)
cat("Թեստի ճշտությունը առանց PCA:", round(score_raw[[2]] * 100, 2), "%\n")
