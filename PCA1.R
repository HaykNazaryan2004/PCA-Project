# Երկու ձևով տվյալների վրա կիրառել ենք նեյրոնային ցանց(Fashion MNIST տվյալների բազայի վրա), որպեսզի հասկանանք PCA-ի ազդեցությունը ուսուցման ժամանակի և ճշտության վրա։ 
# Առաջին կոդում նվազեցրել ենք չափողականությունը  PCA-ի միջոցով հետո կիրառել նեյրոնային ցանց։ Իսկ երկրորդ կոդում միանգամից կիրառել ենք նեյրոնային ցանց։
# Սիդը նվիրված է ԵՊՀ-ի 106 ամյակին
# Fashion MNIST տվյալների բազան պատկերների դասակարգման խնդիրների համար օգտագործվող հանրաճանաչ չափանիշային տվյալների բազա է:
# Այն նախատեսված է որպես MNIST-ի (ձեռագիր թվանշաններ) ավելի իրատեսական այլընտրանք, որը պարունակում է հագուստի մոխրագույն երանգների պատկերներ:


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
c(train_images, train_labels) %<-% fashion_mnist$train # Առանձնացնում է ուսուցանվող տվյալները նկար-իրեն համապատասխան դաս
c(test_images, test_labels) %<-% fashion_mnist$test # նույն գործողությունը՝ այստեղ

# Նկարները դարձնենք միաչափ վեկտորներ (28x28 դարձնենք 784x1)
train_images <- array_reshape(train_images, c(nrow(train_images), 784)) / 255 # այս տողը դարձնում է մեկ չափանի վեկտոր մեր նկարներև և քանի որ պիքսելները արժեք են ընդունում (0,255) բաժանում ենք 255-ի,որպեսզի ընդունի (0,1)  արժեքներ 
test_images <- array_reshape(test_images, c(nrow(test_images), 784)) / 255 # նույն գործողությունը կատարել ենք թեստավորվող տվյալների

#  PCA-ի կիրառում
pca_model <- prcomp(train_images, center = TRUE, scale. = TRUE) #Էստեղ տվյալներից հանել ենք միջինը և բաժանել ենք ստանդարտ շեղման վրա,որը ընդունված պրակտիկա է PCA կիրառելուց առաջ

# Գտնենք այն կոպոնենտների քանակը(գծային տարածության չափողականությունը) որոնք պահանում են 95% վարիացիա
explained_variance <- cumsum(pca_model$sdev^2 / sum(pca_model$sdev^2)) # այս տողը արտածում է թե առաջին n կոմպոնենտները քանի % վարիացիա են բացատրում
num_components <- which(explained_variance >= 0.95)[1] # Սա գտնում է առաջին կոմպոնենտի ինդեքսը, որտեղ գումարային դիսպերսիան ≥ 95% է։
cat("կոմպոնենտների քանակը, որոնք պարունակում են  95% վարիացիա:", num_components, "\n")

# բաժանենք ուսուցանվող և թեստավորվող տվյաներ(train-ի և test-ի բաժանում)
train_pca <- predict(pca_model, train_images)[, 1:num_components] # կիրառում ենք PCA ուսուցանվող տվյալների վրա և պրոյեկտում տվյալները առաջին num_components-ով ենթատարածության վրա։
test_pca <- predict(pca_model, test_images)[, 1:num_components]   # կիրառում ենք PCA թեստավորվող տվյալների վրա և պրոյեկտում տվյալները առաջին num_components-ով ենթատարածության վրա։

# կատերգորիկ փոփոխականները դարձրել ենք թվային
train_labels_cat <- to_categorical(train_labels, 10) 
test_labels_cat <- to_categorical(test_labels, 10)

# կառուցենք նեյրոնային ցանց
model_raw <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(784)) %>% # layer 128 նեյրոնով և relu=max(0, x) activation function-ով(որը դարձնում է մոդելին ոչ գծային)
  layer_dropout(0.3) %>% # նեյրոնների 30%-ը անջատում է,օգնում է վերացնել overfitting-ի դեպքը
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  optimizer = 'adam', # կարգարովորում է learning rate-ը ուսուցման ընթացքում,որպեսզի ավելի արագ զուգամիտի
  loss = 'categorical_crossentropy', # Այն չափում է, թե որքանով է կանխատեսված հավանականությունը հեռու իրականից։
  metrics = 'accuracy' # Ճիշտ կանխատեսումների տոկոսը
)

# չափենք ուսուցման ժամանակը
start_time_pca <- Sys.time()

# ուսուցանել մոդելը
train_history <- model_raw %>% fit(
  x = train_pca, # Սրանք ուսուցման մուտքային feature-ներն են
  y = train_labels_cat, # Սրանք թիրախային label-ներն են՝ one-hot encoded տեսքով
  epochs = 20, # մոդելը անցնում է ամբողջ տվյալների միջով 20 անգամ
  batch_size = 128, # Ուսուցման տվյալները կբաժանվեն 128 նմուշներից բաղկացած խմբերի(375 խումբ),Մոդելը կշիռները թարմացնում է մեկ խմբի համար, այլ ոչ թե յուրաքանչյուր նմուշի համար (սա արագացնում է ուսուցումը)
  validation_split = 0.2, # Ուսուցման տվյալների 20%-ը կօգտագործվի վալիդացիայի համար (ստուգելու համար, թե որքան լավ է մոդելը ընդհանրացնում)։Մոդելը չի ​​ուսուցանվում այս հատվածի վրա, բայց յուրաքանչյուր epoch-ից հետո հաղորդում է դրա ճշգրտության և կորստի մասին։
  verbose = 2
)
training_time_pca <- Sys.time() - start_time_pca
# գնահատենք ճշտությունը տեստ(test)- դատայի վրա
score <- model_raw %>% evaluate(test_pca, test_labels_cat, verbose = 0)
cat("Թեստի ճշտությունը PCA-ով:", round(score[[2]] * 100, 2), "%\n")
cat("Ուսուցման տևողությունը PCA-ով:", training_time_pca, "\n")
