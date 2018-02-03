// basic usage

// load math.js (using node.js)
var math = require('mathjs');
var X =  [ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ];
var y =  transposeMatrix([[0,0,1,1]]);

//var syn0 = randomizeMatrix(makeMatrix(3,4));
//var syn1 = randomizeMatrix(makeMatrix(4,1));

var syn0 = [[0.91850754,-0.207117,0.55211696,0.32125427],
[ 0.10220112,-0.59326735,-0.16234442,0.93764421],
[ 0.28220746,0.22866064,0.1132737,-0.5450892 ]];

var syn1 = [[-0.67699272],[ 0.19260066],[-0.44966636],[-0.13091481]];


for (ite = 0; ite <10000; ite++) {
    var X_syn0 = dotProduct(X,syn0);
    var l1 = expMatrix(productScalarWithMatrix(-1,X_syn0));

    var l1_syn1 = dotProduct(l1,syn1);
    var l2 = expMatrix(productScalarWithMatrix(-1,l1_syn1));
    // l2_delta = (y - l2)*(l2*(1-l2))
    var l2_delta = operationMatrixWithMatrix('*',operationMatrixWithMatrix('-',y,l2),operationMatrixWithMatrix('*',l2,sumScalarToMatrix(1,productScalarWithMatrix(-1,l2))));
    
    //l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    var l1_delta = operationMatrixWithMatrix('*',dotProduct(l2_delta,transposeMatrix(syn1)),operationMatrixWithMatrix('*',l1,sumScalarToMatrix(1,productScalarWithMatrix(-1,l1))));

    //syn1 += l1.T.dot(l2_delta)
    syn1 = operationMatrixWithMatrix('+',syn1,dotProduct(transposeMatrix(l1),l2_delta));

    //syn0 += X.T.dot(l1_delta)
    syn0 = operationMatrixWithMatrix('+',syn0,dotProduct(transposeMatrix(X),l1_delta));

}

console.log(l2);

predict([0,0,1],l1);

function predict(inp, weigths) {
    console.log(inp);
    console.log(weigths);
    //print inp, sigmoid(np.dot(inp, weigths))

}

function expMatrix(mat) {
    var n = mat.length;
    var m = mat[0].length;

    var newMat = makeMatrix(n,m);

    for(i = 0; i < n; i++) {
        for(j = 0; j < m; j++) {
            newMat[i][j] = 1 / (1 + Math.exp(mat[i][j] ));
        }
    }
    return(newMat);
}

function transposeMatrix(mat) {
    var n = mat.length;
    var m = mat[0].length;

    var newMat = [];

    for(j = 0; j < m; j++) {
        var r = [];
        for(i = 0; i < n; i++) {
            r.push(mat[i][j]);
        }  
        newMat.push(r);      
    }
    return(newMat);
}

function randrange(a,b) {
    return Math.floor(Math.random() * (b - a + 1) + a);
 }
 
 function makeMatrix(I, J, fill=0.0) {
    m = [];
    for (i = 0; i < I; i++) {
        r = [];
        for (j = 0; j < J; j++) {
            r.push(fill);
        }
        m.push(r);
    }
    return m;
}

function randomizeMatrix(matrix) {
    var I = matrix.length;
    var J = matrix[0].length;
    for (i = 0; i < I; i++) {
        for (j = 0; j < J; j++) {
            matrix[i][j] = Math.random();
        }
    }
    return matrix;
}

function dotProduct(mat1, mat2) {
    var n1 = mat1.length;
    var m1 = mat1[0].length;

    var n2 = mat2.length;
    var m2 = mat2[0].length;

    var product = makeMatrix(n1,m2);
    
    for(i = 0; i < n1; i++) {
        for(j = 0; j < m2; j++) {
            var s = 0;
            var yyy = '';
            for(k = 0; k < m1; k++) {
                    s += mat1[i][k] * mat2[k][j];
            }  
            product[i][j] = s;
        }  
    }
    return(product);
}

function sumScalarToMatrix(s,mat) {
    var n = mat.length;
    var m = mat[0].length;

    var newMat = makeMatrix(n,m);

    for(i = 0; i < n; i++) {
        for(j = 0; j < m; j++) {
            newMat[i][j] = s + mat[i][j];
        }
    }
    return(newMat);    
}

function productScalarWithMatrix(p,mat) {
    var n = mat.length;
    var m = mat[0].length;

    var newMat = makeMatrix(n,m);

    for(i = 0; i < n; i++) {
        for(j = 0; j < m; j++) {
            newMat[i][j] = p * mat[i][j];
        }
    }
    return(newMat);    
}

function operationMatrixWithMatrix(operation = '+',mat1,mat2) {
    var n = mat1.length;
    var m = mat1[0].length;

    var newMat = makeMatrix(n,m);

    for(i = 0; i < n; i++) {
        for(j = 0; j < m; j++) {
            switch (operation) {
                case '+':
                    newMat[i][j] = mat1[i][j] + mat2[i][j];
                    break;
                case '*':
                    newMat[i][j] = mat1[i][j] * mat2[i][j];
                    break;
                case '-':
                    newMat[i][j] = mat1[i][j] - mat2[i][j];
                    break;
                case '/':
                    newMat[i][j] = mat1[i][j] / mat2[i][j];
                    break;
                default:
                    newMat[i][j] = mat1[i][j] + mat2[i][j];
                    break;
            }
        }
    }
    return(newMat);    
}

/*
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
    l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
    l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
    l2_delta = (y - l2)*(l2*(1-l2))
    l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
    syn1 += l1.T.dot(l2_delta)
    syn0 += X.T.dot(l1_delta)
*/