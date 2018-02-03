
var csv = require("fast-csv");
var fs = require("fs");

var stream = fs.createReadStream("test.csv");
var numColumns = 8;

var dataset = [];
var dataset_min = [];
var dataset_max = [];

for (i = 0; i < numColumns; i++) {
    dataset_min.push(1000.0);
    dataset_max.push(0.0);
}

csv
 .fromStream(stream)
 .transform(function(row){
     newRow = [];
     row.forEach((element,index) => {
         var el = eval(element);
         newRow.push(el);
         if (dataset_min[index] > el) dataset_min[index] = el;
         if (dataset_max[index] < el) dataset_max[index] = el;          
     });
     dataset.push(newRow);         
    })
 .on("data", function(data){
     console.log('data');
 })
 .on("end", function(){
     //console.log(dataset);
     //console.log(dataset_min);
     //console.log(dataset_max);
     var normalized_dataset = normalize_dataset(dataset,dataset_min,dataset_max);
    //console.log(normalized_dataset);
    var n_folds = 5;
    var l_rate = 0.3;
    var n_epoch = 5;
    var n_hidden = 5;

    var scores = evaluate_algorithm(normalized_dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden);
    console.log(scores);
    console.log("done");
 });

 
 function back_propagation(train, test, l_rate, n_epoch, n_hidden) {
     var n_inputs = train[0].length - 1;
     var mySet = [];
     train.forEach(row => {
        if (mySet.indexOf(row[row.length-1]) == -1) {
            mySet.push(row[row.length-1]);
        }
     });
     var n_outputs = mySet.length;
     var network = initialize_network(n_inputs,n_hidden,n_outputs);
    return(network);
 }



 /*
 def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)

 */

 function initialize_network(ni, nh, no) {
    var network = [];
    var hidden_layer = [];
    for (i = 0; i < nh; i++) {
        var list = [];
        for (j = 0; j <= ni; j++) {
            list.push(Math.random());
        }            
        hidden_layer.push({'weights' : list});
    }
    network.push(hidden_layer);
    
    var output_layer = [];
    for (i = 0; i < no; i++) {
        var list = [];
        for (j = 0; j <= nh; j++) {
            list.push(Math.random());
        }            
        output_layer.push({'weights' : list});
    }
    network.push(output_layer);
    
    return(network);
 }

 /*
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    print(network)
    return network

 */

 function normalize_dataset(ds,ds_min,ds_max) {
    ds.forEach((row, i) => {
        var rowLength = row.length;
        row.forEach((element,j) => {
            if (j < rowLength - 1) ds[i][j] = (ds[i][j] - ds_min[j])/(ds_max[j] - ds_min[j]);
        })
    });
    return ds;
 }

 function evaluate_algorithm(ds, al, nf, lr, ne, nh) {
     var folds = cross_validation_split(ds,nf);
     var scores = [];
     folds.forEach((fold,index) => {
        var test_set = fold;
        var train_set = folds.slice(0);
        train_set.splice(index,1);
        train_set = [].concat.apply([],train_set);
        test_set.forEach(row => {
            row[7] = null;
        });
        var predicted = al(train_set,test_set,lr,ne,nh);
        console.log(index, predicted);
    });
     return(null);
 }

 /*
 def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        print('train set',train_set)
        print('test set',test_set)
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

 */

 function cross_validation_split(ds, nf) {
     var ds_split = [];
     var ds_copy = ds;
     var fold_size = Math.floor(ds.length / nf);
     for(i = 0; i < nf; i++) {
        var fold = [];
        while(fold.length < fold_size) {
            var index = randrange(0,ds_copy.length - 1);
            fold.push(ds_copy.pop(index));
        }
        ds_split.push(fold);
     }
     return(ds_split);
 }

 function randrange(a,b) {
    return Math.floor(Math.random() * (b - a + 1) + a);
 }
