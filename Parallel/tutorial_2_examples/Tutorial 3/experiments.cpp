/*

		// 'n'
		std::vector<mytype> inputTestArr(1024, 1);

		// wg size
		size_t local_size = 128;

		size_t padding_size = 0;

		// test for padding (there'll be paddin')
		for (int i = 2; i < inputTestArr.size() * 2; i ^ 2) {
			if (i > inputTestArr.size() && i % local_size == 0) {
				size_t padding_size = i - inputTestArr.size(); // todo make size a variable not a function lol
			}
		}

		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(local_size - padding_size, 0);
			//append that extra vector to our input
			inputTestArr.insert(inputTestArr.end(), A_ext.begin(), A_ext.end());
		}

		size_t input_elements = inputTestArr.size(); //number of input elements
		size_t input_size = inputTestArr.size() * sizeof(mytype); //size of total input in bytes
		size_t nr_groups = input_elements / local_size;

		// end of padding

		int stepCounter = 0;
		int stepLimit = 21; //hardcoded for now, should be log2(padded array size)
		std::vector<mytype> indexer(inputTestArr.size());
		std::vector<mytype> indexer2(inputTestArr.size());
		int padCount = padding_size; // use to subtract padding from padded input vector size
		std::vector<mytype> averages(inputTestArr.size() / 2);
		std::vector<mytype> averages2(inputTestArr.size() / 2);

		std::vector<mytype> summations(inputTestArr.size() / 2);
		// sums of


		// declare buffers
		cl::Kernel kernel_sums = cl::Kernel(program, "get_sums");
		cl::Kernel kernel_avgs = cl::Kernel(program, "get_avgs");
		cl::Kernel kernel_indxs = cl::Kernel(program, "update_indexes");

		cl::Buffer buffer_n(context, CL_MEM_READ_WRITE, input_size);
		cl::Buffer buffer_sumarr(context, CL_MEM_READ_WRITE, (input_size / 2));
		cl::Buffer buffer_indexarr(context, CL_MEM_READ_WRITE, (input_size / 2));
		cl::Buffer buffer_avgarr(context, CL_MEM_READ_WRITE, (input_size / 2));

		cl::Buffer buffer_paddingarr(context, CL_MEM_READ_WRITE, (input_size / 2));

		// enque buffers


		queue.enqueueWriteBuffer(buffer_n, CL_TRUE, 0, input_size, &inputTestArr[0]);
		queue.enqueueFillBuffer(buffer_sumarr, 0, 0, (input_size / 2));
		queue.enqueueFillBuffer(buffer_indexarr, 0, 0, input_size);
		queue.enqueueFillBuffer(buffer_avgarr, 0, 0, (input_size / 2));

		queue.enqueueFillBuffer(buffer_paddingarr, 0, 0, (input_size / 2));


		while (stepCounter < stepLimit) {

			if (stepCounter % 2)
			{
				// calculate sums kernel
					// Summarise every value, return result in sumarr according to
					// that entry's corresponding index value in indexarr
					// also count the padding for each index value and summarise in paddingarr
					// TODO redesign to be structured reductively
					// TOCONSIDER should padding be removed if padding is sorted as though it were
					// a normal integer?
					// additionally, should padding be added where array deviate from
					// expected size?
				kernel_sums.setArg(0, buffer_n);
				kernel_sums.setArg(1, buffer_sumarr);
				kernel_sums.setArg(2, buffer_indexarr);
				kernel_sums.setArg(3, buffer_paddingarr);
				// update avg kernel
					// Using summations of the previous step,
					// calculate new averages to correspond with each index val.
				kernel_avgs.setArg(0, buffer_sumarr);
				kernel_avgs.setArg(1, buffer_paddingarr);
				kernel_avgs.setArg(2, buffer_avgarr);


				// Compare avgs kernel
					// Compare every value of main array to the average array.
					// According to current step of sequence,
					// update the index array (higher vs lower)

					//
					// ??? if compare avg finds a 0 should it skip???
					// what if we pad with unique values that fall
					// outside of the range of the target dataset?
					// Use a scan to find lowest or value, then fill remaining200k values
					// with unique values below or above the dataset's range
					// delete / remove the top / bottom 200k values once sorted
					// !!! in the event of duplicates (val=avgval), indexvalues to one of the
					// index pairs... repeat each time
					// handle duplicates in final step
				kernel_indxs.setArg(0, buffer_n);
				kernel_indxs.setArg(1, buffer_avgarr);
				kernel_indxs.setArg(2, buffer_indexarr);
				kernel_indxs.setArg(3, ); // step counter comes in here
			}

			else {
				// calculate sums kernel
				// update avg2 kernel
				// compare avgs2 kernel
			}

			stepCounter++;


		}
		//update array here
		*/

		/*
				//host - output
				size_t output_size = B.size()*sizeof(mytype);//size in bytes

				//device - buffers
				cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
				cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

				//Part 4 - device operations

				//4.1 copy array A to and initialise other arrays on device memory
				queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &A[0]);
				queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

				//4.2 Setup and execute all kernels (i.e. device code)
				cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_assignment_step_1");
				kernel_1.setArg(0, buffer_A);
				kernel_1.setArg(1, buffer_B);
				kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

				//call all kernels in a sequence
				queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_element_count), cl::NDRange(local_size));


				//4.3 Copy the result from device to host
				queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);
				*/