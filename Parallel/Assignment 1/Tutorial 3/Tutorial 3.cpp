#include "AssignmentFunctions.h"

// Assignment 1; CMP3752M Parallel Programming; Elliott J. Hughes; HUG18683185

/*
Summary:
The majority of the host code has been abstracted into a file called AssignmentFunctions.h. Functions delcared in this file
are called in this file by AF.FunctionName, line 62 contains the declaration of the AF (assginment function) object.

Kernels have been made to handle vector summation, value squaring, vector min/max value searching. Save for the squaring,
these all use a reduction-style approach and have been made sequential as opposed to interleaved in select cases (min & max).

An attempt at sorting has been made, albeit unsuccessfully. An odd-even sort was somewhat successful, but would 'lose' values;
the resulting 'sorted' array would be the same length, have an unsorted 'center', and be missing most frequently the min & max
values. Presumably this was due to overwriting. No amount of memory barriers or kernel redesign appeared to fix the issues.

This program was solely ran and tested on an Nvidia GTX 960M Graphics Card.
*/

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); return 0;}
	}

	// Try detects any potential exceptions
	try {
		// Host operations
		cl::Context context = GetContext(platform_id, device_id);

		// Displays the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		cl::Program::Sources sources;

		AddSources(sources, "kernels/my_kernels.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		// Start of Assignment Code

		AssignmentFunctions AF(context, queue, program);
		
		// Declaring initial variables
		// int vector
		//std::vector<mytype> initialVec;

		// float vector
		std::vector<mytypef> initialVecf;

		// workgroup size
		int local_size = 256;

		// Generic tokeniser + populating initial vector
		char delimiter = ' ';
		std::vector<string> splitText(0);
		std::ifstream txtFile("../../temp_lincolnshire_datasets/temp_lincolnshire_short.txt");
		string text;
		size_t start;
		size_t end;

		// progress iterator 'e'
		int e = 0;
		while (getline(txtFile, text)) {
			start = 0;
			end = 0;
			splitText.clear();

			// Finding the float values
			while (end < text.length() && start < text.length()) {
				end = text.find(delimiter, start);

				if (end == string::npos)
					end = text.length();
				string tok = text.substr(start, end - start);
				if (!tok.empty())
					splitText.push_back(tok);
				start = end + 1;
			}
			// Populating initial vector
			initialVecf.push_back(stof(splitText[5]));

			// Progress reporting
			e++;
			if (!(e % 100000))
				cout << e << " Records Loaded \n";
		}
		txtFile.close();
		int initial_size = initialVecf.size();


		cout << "\n \n \n \nFile Loaded. \n";
		// Test Vector
		// std::vector<mytypef> testVecf{1.0, 3.0, 1.0};

		// Get average
		std::vector<mytypef> Total = AF.callReduceFunctionFloat(initialVecf, "reduce_add_float", local_size);
		float averageVal = Total[0] / initial_size;
		std::cout << "AverageVal = " << averageVal << std::endl;
		std::cout << "AverageValue Kernel Finished in " << AF.timeTaken << " nano seconds.\n" << endl;

		// Get standard deviation
		cl_int sigmoidTime = 0;
		std::vector<mytypef> sigmoidComponentF = AF.callMapFunctionFloat(initialVecf, "map_square_float", local_size, averageVal, 0.001);
		sigmoidTime += AF.timeTaken;
		std::vector<mytypef> sigmoidTotal = AF.callReduceFunctionFloat(sigmoidComponentF, "reduce_add_float", local_size);
		sigmoidTime += AF.timeTaken;

		// Complete standard deviation equation
		float variance = sigmoidTotal[0] / initial_size;
		float sd = sqrt(variance);
		std::cout << "S.D = " << sd << std::endl;
		std::cout << "Standard Deviation Kernels Finished in " << sigmoidTime << " nano seconds.\n" << endl;

		// Get minimum value
		std::vector<mytypef> minInit = AF.callReduceFunctionFloat(initialVecf, "reduce_minimum_float", local_size);
		std::cout << "MinVal = " << minInit << std::endl;
		std::cout << "MinValue Kernel Finished in " << AF.timeTaken << " nano seconds.\n" << endl;
		
		// Get maximum value
		std::vector<mytypef> maxInit = AF.callReduceFunctionFloat(initialVecf, "reduce_maximum_float", local_size);
		std::cout << "MaxVal = " << maxInit << std::endl;
		std::cout << "MaxValue Kernel Finished in " << AF.timeTaken << " nano seconds.\n" << endl;

		std::cout << "All Kernels Finished in " << AF.totalTimeTaken << " nano seconds.\n" << endl;


		//ATTEMPT SORT
		//std::vector<mytypef> sorted = AF.callSortFunctionFloat(initialVecf, "oddeven_assignment_float", local_size);
		//int sortedSize = sorted.size();
		//std::cout << "Median = " << sorted[round(sortedSize * 0.5)] << std::endl;
		//std::cout << "Upper Quartile = " << sorted[(sorted.size() / 4) * 3] << std::endl;
		//std::cout << "Lower Quartile = " << sorted[sorted.size() / 4] << std::endl;
		// TODO rounding on size function
	}

	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}