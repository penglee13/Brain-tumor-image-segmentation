#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "svm.h"
#include <mex.h>
#include "svm_model_matlab.h"

#define BUF_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

// svm arguments
struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;
int cross_validation = 0;
int nr_fold;



void exit_with_help()
{
	mexPrintf(
			"Usage: model = svmtrain(training_label_matrix, training_instance_matrix, 'libsvm_options');\n"
			"libsvm_options:\n"
			"-s svm_type : set type of SVM (default 0)\n"
			"	0 -- C-SVC\n"
			"	1 -- nu-SVC\n"
			"	2 -- one-class SVM\n"
			"	3 -- epsilon-SVR\n"
			"	4 -- nu-SVR\n"
			"-t kernel_type : set type of kernel function (default 2)\n"
			"	0 -- linear: u'*v\n"
			"	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
			"	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
			"	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
			"-d degree : set degree in kernel function (default 3)\n"
			"-g gamma : set gamma in kernel function (default 1/k)\n"
			"-r coef0 : set coef0 in kernel function (default 0)\n"
			"-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
			"-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
			"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
			"-m cachesize : set cache memory size in MB (default 40)\n"
			"-e epsilon : set tolerance of termination criterion (default 0.001)\n"
			"-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
			"-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
			"-wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
			"-v n: n-fold cross validation mode\n"
			);
}

double do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
			param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		mexPrintf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		mexPrintf("Cross Validation Squared correlation coefficient = %g\n",
				((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
				((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
				);
		return total_error/prob.l;
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		mexPrintf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
		return 100.0*total_correct/prob.l;
	}
	free(target);
}

// nrhs should be 3
int parse_command_line(int nrhs, const mxArray *prhs[], char *model_file_name)
{
	int i, flag = 0;
	char cmd[BUF_LEN];
	char *argv_1, *argv_2;

	// default values
	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/k
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 40;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	cross_validation = 0;
	
	// parse options
	if(nrhs < 3) 
		return(0);	

	/* Get options */
	mxGetString(prhs[2], cmd,  mxGetN(prhs[2]) + 1);
	while(1) 
	{	
		if(flag == 0) 
		{
			argv_1 = strtok(cmd, " ");	
			flag = 1;
		}
		else 
		{
			argv_1 = strtok(NULL, " ");
		}

		if(argv_1 == NULL || argv_1[0] != '-') break;
		++i;

		argv_2 = strtok(NULL, " ");
		if(argv_2 == NULL) 
		{
			exit_with_help();
			return(1);	
		}

		switch(argv_1[1])
		{
		case 's':
			param.svm_type = atoi(argv_2);
			break;
		case 't':
			param.kernel_type = atoi(argv_2);
			break;
		case 'd':
			param.degree = atof(argv_2);
			break;
		case 'g':
			param.gamma = atof(argv_2);
			break;
		case 'r':
			param.coef0 = atof(argv_2);
			break;
		case 'n':
			param.nu = atof(argv_2);
			break;
		case 'm':
			param.cache_size = atof(argv_2);
			break;
		case 'c':
			param.C = atof(argv_2);
			break;
		case 'e':
			param.eps = atof(argv_2);
			break;
		case 'p':
			param.p = atof(argv_2);
			break;
		case 'h':
			param.shrinking = atoi(argv_2);
			break;
		case 'b':
			param.probability = atoi(argv_2);
			break;
		case 'v':
			cross_validation = 1;
			nr_fold = atoi(argv_2);
			if(nr_fold < 2)
			{
				mexPrintf("n-fold cross validation: n must >= 2\n");
				return(1);
			}
			break;
		case 'w':
			++param.nr_weight;
			param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
			param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
			param.weight_label[param.nr_weight-1] = atoi(&argv_1[2]);
			param.weight[param.nr_weight-1] = atof(argv_2);
			break;
		default:
			mexPrintf("unknown option\n");
			return(1);
		}
	}
}

// read in a problem (in svmlight format)
void read_problem_dense(double *labels, double *samples, int lr, int lc, int sr, int sc)
{
	int elements, max_index, i, j, k;

	elements = 0;
	// the number of instance
	prob.l = sr;
	for(i = 0; i < prob.l; i++) 
	{
		for(k = 0; k < sc; k++) 
		{
			if(samples[k * prob.l + i] != 0) 
			{
				elements++;
			}	
		}
		// count the '-1' element
		elements++;
	}

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = sc;
	j = 0;
	for(i = 0; i < prob.l; i++)
	{
		prob.x[i] = &x_space[j];
		prob.y[i] = labels[i];					
		for(k = 0; k < sc; k++) 
		{
			if(samples[k * prob.l + i] != 0) 
			{
				x_space[j].index = k + 1;
				x_space[j].value = samples[k * prob.l + i];
				++j;	
			}	
		}
		x_space[j++].index = -1;
	}
	if(param.gamma == 0)
		param.gamma = 1.0/max_index;
		
}


void read_problem_sparse(const mxArray *prhs[])
{
	int elements, max_index, i, j, k;
	int sr, sc, lr, lc;
	int *ir, *jc, *tmp_row_index;
	int tmp_index_sum, tmp_jc_num, pr_ir_index;
	int num_samples, num_ir, num_jc;
	double *samples, *labels;

	lr = mxGetM(prhs[0]);
	lc = mxGetN(prhs[0]);
	sr = mxGetM(prhs[1]);
	sc = mxGetN(prhs[1]);

	labels = mxGetPr(prhs[0]);
	samples = mxGetPr(prhs[1]);
	ir = mxGetIr(prhs[1]);
	jc = mxGetJc(prhs[1]);

	num_jc = sc + 1;
	num_samples = num_ir = jc[num_jc-1];

	tmp_row_index = (int *)calloc(sr, sizeof(int));

	// the number of instance
	prob.l = sr;
	elements = num_samples + prob.l;

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node, elements);

	max_index = sc;
	for(i = 0; i < num_ir; i++)
		tmp_row_index[ir[i]]++;

	for(i = tmp_index_sum = 0; i < prob.l; i++) {
		prob.x[i] = &x_space[tmp_index_sum];
		prob.y[i] = labels[i];	
		tmp_index_sum += (tmp_row_index[i] + 1);
		x_space[tmp_index_sum - 1].index = -1;
	}

	pr_ir_index = 0;
	memset(tmp_row_index, 0, sr * sizeof(int));
	for(i = 1; i < num_jc; i++)
	{
		tmp_jc_num = jc[i] - jc[i-1];
		for(j = 0; j < tmp_jc_num; j++)
		{
			prob.x[ir[pr_ir_index]][tmp_row_index[ir[pr_ir_index]]].index = i;
			prob.x[ir[pr_ir_index]][tmp_row_index[ir[pr_ir_index]]].value = samples[pr_ir_index];
			tmp_row_index[ir[pr_ir_index]]++;
			pr_ir_index++;
		}
	}

	if(param.gamma == 0)
		param.gamma = 1.0/max_index;	

	free(tmp_row_index);
}

// Interface function of matlab
// now assume prhs[0]: label prhs[1]: features
void mexFunction( int nlhs, mxArray *plhs[], 
		int nrhs, const mxArray *prhs[] )
{ 
	int m1, n1, m2, n2, i, j;
	double *labels, *samples, *ptr;
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	// Translate the input Matrix to the format such that svmtrain.exe can recognize it
	if(nrhs > 0 && nrhs < 4) 
	{
		if (parse_command_line(nrhs, prhs, model_file_name)==1)
		{
			plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
			return;
		}
			
		m1 = mxGetM(prhs[0]);
		n1 = mxGetN(prhs[0]);
		m2 = mxGetM(prhs[1]);
		n2 = mxGetN(prhs[1]);
		
		if(mxIsSparse(prhs[1]))
			read_problem_sparse(prhs);
		else 
		{
			labels = mxGetPr(prhs[0]);
			samples = mxGetPr(prhs[1]);
			read_problem_dense(labels, samples, m1, n1, m2, n2);
		}

		// svmtrain's original code
		error_msg = svm_check_parameter(&prob, &param);

		if(error_msg)
		{
			mexPrintf("Error: %s\n", error_msg);
			plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
			return;
		}

		if(cross_validation)
		{
			plhs[0] = mxCreateDoubleMatrix(1, 1, mxREAL);
			ptr = mxGetPr(plhs[0]);
			ptr[0] = do_cross_validation();
		}
		else
		{
			model = svm_train(&prob, &param);
			model_to_matlab_structure(plhs, n2, model);
			svm_destroy_model(model);
		}
		svm_destroy_param(&param);
		free(prob.y);
		free(prob.x);
		free(x_space);
	}
	else 
	{
		exit_with_help();
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
	}
	return;
}
