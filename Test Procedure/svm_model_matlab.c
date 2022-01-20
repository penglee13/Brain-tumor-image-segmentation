#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include "svm.h"


#define NUM_OF_RETURN_FIELD 10

void model_to_matlab_structure(mxArray *plhs[], int num_of_feature, struct svm_model *model){

	int i, j, n, *ir, *jc, *tmp_row_index, ir_index, jc_index, ptr_index;
	double *ptr;
	mxArray *cell_array_ptr, **rhs;
	int out_id = 0, nonzero_row, nonzero_element;

	rhs = (mxArray **)mxMalloc(sizeof(mxArray *)*NUM_OF_RETURN_FIELD);	

	// Parameters
	rhs[out_id] = mxCreateDoubleMatrix(5, 1, mxREAL); 
	ptr = mxGetPr(rhs[out_id]);
	ptr[0] = model->param.svm_type;
	ptr[1] = model->param.kernel_type;
	ptr[2] = model->param.degree;
	ptr[3] = model->param.gamma;
	ptr[4] = model->param.coef0;
	out_id++;

	// nr_class
	rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	ptr[0] = model->nr_class;
	out_id++;
	
	// total SV
	rhs[out_id] = mxCreateDoubleMatrix(1, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	ptr[0] = model->l;
	out_id++;

	n = model->nr_class*(model->nr_class-1)/2;
	// rho
	rhs[out_id] = mxCreateDoubleMatrix(n, 1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	for(i = 0; i < n; i++)
		ptr[i] = model->rho[i];
	out_id++;

	// Label
	if(model->label)
	{
		rhs[out_id] = mxCreateDoubleMatrix(model->nr_class, 1, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		for(i = 0; i < model->nr_class; i++)
			ptr[i] = model->label[i];
	}
	else
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	out_id++;

	// probA, probB
	if(model->probA != NULL && model->probB != NULL)
	{
		rhs[out_id] = mxCreateDoubleMatrix(n, 1, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		for(i = 0; i < n; i++)
			ptr[i] = model->probA[i];

		rhs[out_id+1] = mxCreateDoubleMatrix(n, 1, mxREAL);
		ptr = mxGetPr(rhs[out_id+1]);
		for(i = 0; i < n; i++)
			ptr[i] = model->probB[i];
	}
	else 
	{
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
		rhs[out_id+1] = mxCreateDoubleMatrix(0, 0, mxREAL);
	}
	out_id+=2;

	// nSV
	if(model->nSV)
	{
		rhs[out_id] = mxCreateDoubleMatrix(model->nr_class, 1, mxREAL);
		ptr = mxGetPr(rhs[out_id]);
		for(i = 0; i < model->nr_class; i++)
			ptr[i] = model->nSV[i];
	}
	else
		rhs[out_id] = mxCreateDoubleMatrix(0, 0, mxREAL);
	out_id++;

	// sv_coef
	rhs[out_id] = mxCreateDoubleMatrix(model->l, model->nr_class-1, mxREAL);
	ptr = mxGetPr(rhs[out_id]);
	for(i = 0; i < model->nr_class-1; i++)
		for(j = 0; j < model->l; j++)
			ptr[(i*(model->l))+j] = model->sv_coef[i][j];
	out_id++;

	// SVs
	nonzero_element = 0;
	for(i = 0; i < model->l; i++) {
		j = 0;
		while(model->SV[i][j].index != -1) {
			nonzero_element++;
			j++;
		}
	}
	
	rhs[out_id] = mxCreateSparse(model->l, num_of_feature, nonzero_element, mxREAL);
	ir = mxGetIr(rhs[out_id]);
	jc = mxGetJc(rhs[out_id]);
	ptr = mxGetPr(rhs[out_id]);
	ir_index = ptr_index = jc_index = 0;
	jc[jc_index++] = 0;
	tmp_row_index = (int *)calloc(model->l, sizeof(int));
	memset(tmp_row_index, 0, model->l * sizeof(int));
	for(i = 0; i < num_of_feature; i++)
	{	
		nonzero_row = 0;
		for(j = 0; j < model->l; j++)
		{
			if(model->SV[j][tmp_row_index[j]].index == (i + 1))
			{
				ptr[ptr_index++] = model->SV[j][tmp_row_index[j]].value;
				ir[ir_index++] = j;
				nonzero_row++;
				tmp_row_index[j]++;
			}
		}
		jc[jc_index] = jc[jc_index-1] + nonzero_row;
		jc_index++;
	}	
	free(tmp_row_index);
	out_id++;

	/* Create a nrhs x 1 cell mxArray. */ 
	cell_array_ptr = mxCreateCellMatrix(NUM_OF_RETURN_FIELD, 1);

	/* Fill cell matrix with input arguments */
	for(i = 0; i < NUM_OF_RETURN_FIELD; i++)
	{
		mxSetCell(cell_array_ptr,i,mxDuplicateArray(rhs[i]));
	}
	/* return */
	plhs[0] = cell_array_ptr;

	return ;
}
