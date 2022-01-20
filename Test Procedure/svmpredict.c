#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mex.h"
#include "svm.h"
#include "svm_model_matlab.h"

struct svm_node *x_space;

void exit_with_help()
{
	mexPrintf(
	"Usage: [predict_label, accuracy] = svmpredict(testing_label_matrix, testing_instance_matrix, model, 'libsvm_options')\n"
	"libsvm_options:\n"
	"-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet\n"
	);
}

void matlab_matrix_to_model(struct svm_model *model, mxArray **input_array)
{
	int mrows, ncols;		
	int total_l, feature_number;
	int i, j, n;
	double *ptr;
	bool bpredict_probability;
	int id = 0;

	model->rho = NULL;
	model->probA = NULL;
	model->probB = NULL;
	model->label = NULL;
	model->nSV = NULL;

	ptr = mxGetPr(input_array[id]);
	model->param.svm_type	  = (int)ptr[0];
	model->param.kernel_type  = (int)ptr[1];
	model->param.degree	  = ptr[2];
	model->param.gamma	  = ptr[3];
	model->param.coef0	  = ptr[4];
	id++;

	ptr = mxGetPr(input_array[id]);
	model->nr_class = (int)ptr[0];
	id++;

	ptr = mxGetPr(input_array[id]);
	model->l = (int)ptr[0];
	id++;

	// rho
	n = model->nr_class * (model->nr_class-1)/2;
	model->rho = (double*) malloc(n*sizeof(double));
	ptr = mxGetPr(input_array[id]);
	for(i=0;i<n;i++)
		model->rho[i] = ptr[i];
	id++;

	// label
	if (mxIsEmpty(input_array[id]) == 0)
	{
		model->label = (int*) malloc(model->nr_class*sizeof(int));
		ptr = mxGetPr(input_array[id]);
		for(i=0;i<model->nr_class;i++)
			model->label[i] = (int)ptr[i];
	}
	id++;

	// probA, probB
	if(mxIsEmpty(input_array[id]) == 0 && 
	   mxIsEmpty(input_array[id+1]) == 0)
	{
		model->probA = (double*) malloc(n*sizeof(double));
		model->probB = (double*) malloc(n*sizeof(double));
		ptr = mxGetPr(input_array[id]);
		for(i=0;i<n;i++)
			model->probA[i] = ptr[i];
		ptr = mxGetPr(input_array[id+1]);
		for(i=0;i<n;i++)
			model->probB[i] = ptr[i];
	}
	id += 2;

	// nSV
	if (mxIsEmpty(input_array[id]) == 0)
	{
		model->nSV = (int*) malloc(model->nr_class*sizeof(int));
		ptr = mxGetPr(input_array[id]);
		for(i=0;i<model->nr_class;i++)
			model->nSV[i] = (int)ptr[i];
	}
	id++;

	// sv_coef
	ptr = mxGetPr(input_array[id]);
	model->sv_coef = (double**) malloc((model->nr_class-1)*sizeof(double));
	for( i=0 ; i< model->nr_class -1 ; i++ )
		model->sv_coef[i] = (double*) malloc((model->l)*sizeof(double));
	for(i = 0; i < model->nr_class - 1; i++)
		for(j = 0; j < model->l; j++)
			model->sv_coef[i][j] = ptr[i*(model->l)+j];
	id++;

	// SV		
	{	
		int sr, sc, elements;
		int num_samples, num_ir, num_jc;
		int *ir, *jc, *tmp_row_index;	
		int tmp_index_sum, tmp_jc_num, pr_ir_index;
		
		sr = mxGetM(input_array[id]);
		sc = mxGetN(input_array[id]);
		
		ptr = mxGetPr(input_array[id]);
		ir = mxGetIr(input_array[id]);
		jc = mxGetJc(input_array[id]);
		
		num_jc = sc + 1;
		num_samples = num_ir = jc[num_jc-1];
		
		elements = num_samples + sr;
		
		model->SV = (struct svm_node **) malloc(sr * sizeof(struct svm_node *));
		x_space = (struct svm_node *)malloc(elements * sizeof(struct svm_node));
		tmp_row_index = (int *)malloc(sr * sizeof(int));
		
		memset(tmp_row_index, 0, sr * sizeof(int));
		for(i = 0; i < num_ir; i++)
			tmp_row_index[ir[i]]++;

		for(i = tmp_index_sum = 0; i < sr; i++) {
			model->SV[i] = &x_space[tmp_index_sum];	
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
				model->SV[ir[pr_ir_index]][tmp_row_index[ir[pr_ir_index]]].index = i;
				model->SV[ir[pr_ir_index]][tmp_row_index[ir[pr_ir_index]]].value = ptr[pr_ir_index];
				tmp_row_index[ir[pr_ir_index]]++;
				pr_ir_index++;
			}
		}
		free(tmp_row_index);
		id++;
	}
}

void read_sparse_instance(const mxArray *prhs[], int index, struct svm_node *x)
{
	int sr, sc, *ir, *jc, i, j;
	int num_samples, num_ir, num_jc, x_index;
	int low, mid, high;
	double *samples;

	sr = mxGetM(prhs[1]);
	sc = mxGetN(prhs[1]);

	samples = mxGetPr(prhs[1]);
	ir = mxGetIr(prhs[1]);
	jc = mxGetJc(prhs[1]);

	num_samples = num_ir = mxGetNzmax(prhs[1]);
	num_jc = sc + 1;
	
	for(i = x_index = 0; i < num_jc - 1; i++)
	{
		low = jc[i], high = jc[i+1] - 1, mid = (low+high)/2;
		while(low <= high) 
		{
			if(ir[mid] == index) 
			{
				x[x_index].index = i + 1;
				x[x_index].value = samples[mid];
				x_index++;
				break;
			}
			else if(ir[mid] > index)
				high = mid - 1;
			else
				low = mid + 1;

			mid = (low+high)/2;
		}
	}
	x[x_index].index = -1;

}



void predict(mxArray *plhs[], const mxArray *prhs[], struct svm_model *model, const int predict_probability)
{
	int instance_matrix_row_num, instance_matrix_col_num;
	int type_matrix_row_num, type_matrix_col_num;
	int feature_number, testing_instance_number;
	int i, j, instance_index;
	int matrix_width;
	double *ptr_instance, *ptr_type, *ptr_predict_type, *ptr;
	struct svm_node *x;
	
	int correct = 0;
	int total = 0;
	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double target,v;
	double *prob_estimates=NULL;
	int *labels=(int *) malloc(nr_class*sizeof(int));

	//prhs[1] = testing instance vectors
	feature_number = instance_matrix_col_num = mxGetN(prhs[1]);
	testing_instance_number = instance_matrix_row_num = mxGetM(prhs[1]);
	type_matrix_row_num = mxGetM(prhs[0]);
	type_matrix_col_num = mxGetN(prhs[0]);
	
	if(type_matrix_row_num!=instance_matrix_row_num)
	{
		mexPrintf("prhs[1](Instance)'s row number should be the same as prhs[0](Type)'s row number, but not.\n");
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		return;						
	}
	if(type_matrix_col_num!=1)
	{
		mexPrintf("prhs[0](Type)'s col number should be 1, but not.\n");
		plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
		return;						
	}
	
	ptr_instance = mxGetPr(prhs[1]);
	ptr_type     = mxGetPr(prhs[0]);

	if(predict_probability)
	{
		if(svm_type==NU_SVR || svm_type==EPSILON_SVR)
			mexPrintf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
	}
	
	if(predict_probability && (svm_type==C_SVC || svm_type==NU_SVC)) 
		plhs[0] = mxCreateDoubleMatrix(testing_instance_number, 1+nr_class, mxREAL);
	else
		plhs[0] = mxCreateDoubleMatrix(testing_instance_number, 1, mxREAL);
	
	ptr_predict_type = mxGetPr(plhs[0]);
	x = (struct svm_node*)malloc((feature_number+1)*sizeof(struct svm_node) );	
	for(instance_index=0;instance_index<testing_instance_number;instance_index++)
	{
		target = ptr_type[instance_index];

		if(mxIsSparse(prhs[1]))
			read_sparse_instance(prhs, instance_index, x);	
		else
		{
			for(i=0;i<feature_number;i++)
			{
				x[i].index = i+1;
				x[i].value = ptr_instance[testing_instance_number*i+instance_index];
			}
			x[feature_number].index = -1;
		}

		if(predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			v = svm_predict_probability(model, x, prob_estimates);		
			ptr_predict_type[instance_index] = v;
			for(i=1;i<=nr_class;i++)
				ptr_predict_type[instance_index + i * testing_instance_number] = prob_estimates[i-1];
		}
		else
		{
			v = svm_predict(model,x);
			ptr_predict_type[instance_index] = v;
		}
			
		if(v == target)
			++correct;
		error += (v-target)*(v-target);
		sumv += v;
		sumy += target;
		sumvv += v*v;
		sumyy += target*target;
		sumvy += v*target;
		++total;
	}

	mexPrintf("Accuracy = %g%% (%d/%d) (classification)\n",
			(double)correct/total*100,correct,total);
	mexPrintf("Mean squared error = %g (regression)\n",error/total);
	mexPrintf("Squared correlation coefficient = %g (regression)\n",
		((total*sumvy-sumv*sumy)*(total*sumvy-sumv*sumy))/
		((total*sumvv-sumv*sumv)*(total*sumyy-sumy*sumy))
		 );

	// return accuracy, mean squared error, squared correlation coefficient
	plhs[1] = mxCreateDoubleMatrix(3, 1, mxREAL);
	ptr = mxGetPr(plhs[1]);
	ptr[0] = (double)correct/total*100;
	ptr[1] = error/total;
	ptr[2] = ((total*sumvy-sumv*sumy)*(total*sumvy-sumv*sumy))/
				((total*sumvv-sumv*sumv)*(total*sumyy-sumy*sumy));
	
	free(x);
	if(predict_probability)
	{
		if(prob_estimates != NULL)
			free(prob_estimates);
		free(labels);
	}
	
}

void mexFunction( int nlhs, mxArray *plhs[], 
		 int nrhs, const mxArray *prhs[] )
{
	int i, num_of_cell_elements, prob_estimate_flag;
	struct svm_model *model;
	mxArray **rhs;
	char option[2048], *tmp_option;

	model = (struct svm_model *) malloc(sizeof(struct svm_model));

	if(nrhs > 4 || nrhs < 3){
		exit_with_help();
		return;
	}
	
	if(mxIsCell(prhs[2])) 
	{		
		num_of_cell_elements = mxGetNumberOfElements(prhs[2]);
		rhs = (mxArray **) malloc(sizeof(mxArray *)*num_of_cell_elements);
		
		for(i=0;i<num_of_cell_elements;i++)
			rhs[i] = mxGetCell(prhs[2], i);
		
		matlab_matrix_to_model(model, rhs);
		
		prob_estimate_flag = 0;
		// option -b		
		if(nrhs==4) 
		{
			mxGetString(prhs[3], option, mxGetN(prhs[3])+1);
			tmp_option = strtok(option, " ");
			
			if(strcmp(tmp_option, "-b")) 
			{
				mexPrintf("The option should be -b\n");
				return;	
			}
			
			tmp_option = strtok(NULL, " ");
			prob_estimate_flag = atoi(tmp_option);
		
			if(prob_estimate_flag==1 && (model->probA==NULL || model->probB==NULL)) 
			{
				mexPrintf("There is no probability estimate model in the model file.\n");
				plhs[0] = mxCreateDoubleMatrix(0, 0, mxREAL);
				return;
			}
		}

		predict(plhs, prhs, model, prob_estimate_flag);
		// destroy model
		svm_destroy_model(model);
	}
	else 
		mexPrintf("model file should be a cell array\n");	
	
	return;
}
