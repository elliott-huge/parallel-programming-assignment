#pragma once
#include <iostream>
#include <vector>
#include <fstream>

#include "Utils.h"

// int and float custom types
typedef int mytype;
typedef float mytypef;

class AssignmentFunctions
{

private:
	cl::Context contx;
	cl::CommandQueue que;
	cl::Program prog;



public:
	// reset at the beginning of every operation
	cl_int timeTaken;

	// sums all kernel run times for a total
	cl_int totalTimeTaken;

	// common variables
	AssignmentFunctions(cl::Context c, cl::CommandQueue q, cl::Program p)
	{
		contx = c;
		que = q;
		prog = p;
		totalTimeTaken = 0;
	}



	// functions that handle reduce and map operations
	std::vector<mytypef> callReduceFunctionFloat(std::vector<mytypef> input, char* kernelName, int wgSize, cl_float optParam = 0.0) {
		std::vector<mytypef> A = input;
		size_t local_size = wgSize;
		int initial_size = A.size();
		size_t padding_size;
		timeTaken = 0;
		cl_int partialTimeTaken = 0;

		// B vector
		std::vector<mytypef> B(2, 0.0);

		// this structure may be reused for different reduce, operations
		// the only thing that needs to change is the kernel's "name" parameter
		while (B.size() > 1)
		{
			// handle padding

			padding_size = initial_size % local_size;
			if (padding_size) {
				std::vector<float> A_ext(local_size - padding_size, 0.0);
				A.insert(A.end(), A_ext.begin(), A_ext.end());
			}
			// set sizing
			size_t input_element_count = A.size();
			size_t input_size = A.size() * sizeof(mytypef);
			size_t nr_groups = input_element_count / local_size;

			// resize & zero vector B (update nr_groups)
			B.resize(nr_groups, 0.0);
			B.shrink_to_fit();

			size_t output_size = B.size() * sizeof(mytypef);

			//set buffers
			cl::Buffer buffer_A(contx, CL_MEM_READ_ONLY, input_size);
			cl::Buffer buffer_B(contx, CL_MEM_READ_WRITE, output_size);
			que.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
			que.enqueueFillBuffer(buffer_B, 0, 0, output_size);

			//run kernels
			cl::Kernel kernel_1 = cl::Kernel(prog, kernelName);
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(local_size * sizeof(mytypef)));
			if (optParam)
				kernel_1.setArg(3, optParam);

			cl::Event prof_event;

			que.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_element_count), cl::NDRange(local_size), NULL, &prof_event);

			//retrieve output
			que.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
			partialTimeTaken = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
			timeTaken += partialTimeTaken;
			
			// then assign vector B's values to vector A after resizing
			A.resize(B.size());
			A.shrink_to_fit();

			A = B;

			// Returns profiling information
			std::cout << kernelName << ": " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;
		}
		
		totalTimeTaken += timeTaken;

		return B;
	}
	std::vector<mytypef> callMapFunctionFloat(std::vector<mytypef> input, char* kernelName, int wgSize, cl_float optParam = 0.0, float padVal = 0.00001) {

		std::vector<mytypef> initialVec = input;
		std::vector<mytypef> B;
		timeTaken = 0;
		int local_size = wgSize;
		int padding_size;
		padding_size = initialVec.size() % local_size;
		if (padding_size) {
			std::vector<float> i_ext(local_size - padding_size, padVal);
			initialVec.insert(initialVec.end(), i_ext.begin(), i_ext.end());
		}

		size_t input_element_count = initialVec.size();
		size_t input_size = initialVec.size() * sizeof(mytypef);

		// resize & zero vector B (update nr_groups)
		B.resize(input_element_count, 0);
		B.shrink_to_fit();

		size_t output_size = B.size() * sizeof(mytypef);

		//set buffers
		cl::Buffer buffer_A(contx, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(contx, CL_MEM_READ_WRITE, output_size);
		que.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &initialVec[0]);
		que.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		//run kernels
		cl::Kernel kernel_1 = cl::Kernel(prog, kernelName);
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, optParam);
		kernel_1.setArg(3, padVal);

		cl::Event prof_event;

		que.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_element_count), cl::NDRange(local_size), NULL, &prof_event);

		//retrieve output
		que.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		timeTaken = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		totalTimeTaken += timeTaken;

		std::cout << kernelName << ": " << GetFullProfilingInfo(prof_event, ProfilingResolution::PROF_US) << std::endl;

		return B;
	}
	




	// #### EVERYTHING BELOW HERE IS UNUSED ####

	// int versions
	// function to handle reduce operations (integer)
	std::vector<mytype> callReduceFunction(std::vector<mytype> input, char* kernelName, int wgSize, cl_int optParam = 0) {
		std::vector<mytype> A = input;
		size_t local_size = wgSize;
		int initial_size = A.size();
		size_t padding_size = A.size() % local_size;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			A.insert(A.end(), A_ext.begin(), A_ext.end());
		}

		// B vector
		std::vector<mytype> B(2, 0);

		// this structure may be reused for different reduce, operations
		// the only thing that needs to change is the kernel's "name" parameter
		while (B.size() > 1)
		{
			// set sizing
			size_t input_element_count = A.size();
			size_t input_size = A.size() * sizeof(mytype);
			size_t nr_groups = input_element_count / local_size;

			// resize & zero vector B (update nr_groups)
			B.resize(nr_groups, 0);
			B.shrink_to_fit();

			size_t output_size = B.size() * sizeof(mytype);

			//set buffers
			cl::Buffer buffer_A(contx, CL_MEM_READ_ONLY, input_size);
			cl::Buffer buffer_B(contx, CL_MEM_READ_WRITE, output_size);
			que.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
			que.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

			//run kernels
			cl::Kernel kernel_1 = cl::Kernel(prog, kernelName);
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(local_size * sizeof(mytype)));
			if (optParam)
				kernel_1.setArg(3, optParam);

			que.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_element_count), cl::NDRange(local_size));

			//retrieve output
			que.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

			// then assign vector B's values to vector A after resizing
			A.resize(B.size());
			A.shrink_to_fit();

			A = B;
			// looks awful but this is passed by value so it works A-ok

			// repadd if necessary, otherwise the kernels will fret over buffer-size
			padding_size = A.size() % local_size;
			if (padding_size) {
				std::vector<int> A_ext(local_size - padding_size, 0);
				A.insert(A.end(), A_ext.begin(), A_ext.end());
			}
		}
		return B;
	}
	// function to handle map operations (integer)
	std::vector<mytype> callMapFunction(std::vector<mytype> input, char* kernelName, int wgSize, cl_int optParam = 0) {

		std::vector<mytype> initialVec = input;
		std::vector<mytype> B;
		timeTaken = 0;
		int local_size = wgSize;
		int padding_size;
		padding_size = initialVec.size() % local_size;
		if (padding_size) {
			std::vector<int> i_ext(local_size - padding_size, 0);
			initialVec.insert(initialVec.end(), i_ext.begin(), i_ext.end());
		}

		size_t input_element_count = initialVec.size();
		size_t input_size = initialVec.size() * sizeof(mytype);

		// resize & zero vector B (update nr_groups)
		B.resize(input_element_count, 0);
		B.shrink_to_fit();

		size_t output_size = B.size() * sizeof(mytype);

		//set buffers
		cl::Buffer buffer_A(contx, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(contx, CL_MEM_READ_WRITE, output_size);
		que.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &initialVec[0]);
		que.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

		//run kernels
		cl::Kernel kernel_1 = cl::Kernel(prog, kernelName);
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, optParam);

		cl::Event prof_event;

		que.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_element_count), cl::NDRange(local_size), NULL, &prof_event);

		//retrieve output
		que.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

		timeTaken = prof_event.getProfilingInfo<CL_PROFILING_COMMAND_END>() - prof_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		totalTimeTaken += timeTaken;
		return B;
	}



	// unused attempts:
		// function that attempts to handle sort operation
	std::vector<mytypef> callSortFunctionFloat(std::vector<mytypef> input, char* kernelName, int wgSize, cl_float optParam = 0.0, float padVal = 0.00001) {

		std::vector<mytypef> initialVec = input;
		int local_size = wgSize;



		size_t input_element_count = initialVec.size();
		size_t input_size = input_element_count * sizeof(mytypef);
		std::vector<mytypef> B(input_element_count);

		//set buffers
		cl::Buffer buffer_A(contx, CL_MEM_READ_WRITE, input_size);
		que.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &initialVec[0]);

		//run kernels
		cl::Kernel kernel_1 = cl::Kernel(prog, kernelName);
		kernel_1.setArg(0, buffer_A);

		do
		{
			que.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_element_count), cl::NullRange);

			//retrieve output
			que.enqueueReadBuffer(buffer_A, CL_TRUE, 0, input_size, &B[0]);
		} while (!std::is_sorted(B.begin(), B.end()));

		/*
		int startRm = -1;
		int endRm = -1;
		for (int i = 0; i < B.size(); i++)
		{
			if (B[i] == padVal && startRm == -1)
				startRm = i;
			else if (B[i] == padVal)
				endRm = i;
		}
		cout << startRm << " " << endRm << endl;
		*/
		//B.erase(startRm, endRm);

		return B;
	}


	// second attempt using reduce style sorting
	std::vector<mytypef> callSortFunctionFloatTwo(std::vector<mytypef> input, int wgSize, cl_float optParam = 0.0, float padVal = 0.00001) {

		std::vector<mytypef> initialVec = input;
		int local_size = wgSize;
		int padding_size;
		padding_size = initialVec.size() % local_size;
		if (padding_size) {
			std::vector<float> i_ext(local_size - padding_size, 0.0);
			initialVec.insert(initialVec.end(), i_ext.begin(), i_ext.end());
		}
		// set sizing
		size_t input_element_count = initialVec.size();
		size_t input_size = initialVec.size() * sizeof(mytypef);



		size_t nr_groups = input_element_count / local_size;
		size_t output_size = nr_groups * sizeof(mytypef);
		std::vector<mytypef> B(nr_groups);

		//set buffers
		cl::Buffer buffer_A(contx, CL_MEM_READ_ONLY, input_size);
		cl::Buffer buffer_B(contx, CL_MEM_READ_WRITE, output_size);
		que.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &initialVec[0]);
		que.enqueueFillBuffer(buffer_B, 0, 0, input_size);
		cl_int iter = 0;
		//run kernels
		cl::Kernel kernel_1 = cl::Kernel(prog, "sort_by_minimum_reduce");
		kernel_1.setArg(0, buffer_A);
		kernel_1.setArg(1, buffer_B);
		kernel_1.setArg(2, cl::Local(local_size * sizeof(mytypef)));
		kernel_1.setArg(3, iter);


		for (int i = 0; i < input_element_count; i++)
		{
			que.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_element_count), cl::NDRange(local_size));
		}
		que.enqueueReadBuffer(buffer_B, CL_TRUE, 0, input_size, &B[0]);

		// remove padding here
		cl::Kernel kernel_2 = cl::Kernel(prog, "");
		kernel_2.setArg(0, padVal);

		return B;
	}



};

