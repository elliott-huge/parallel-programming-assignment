#include "AssignmentFunctions.h"

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

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Running on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context);

		//2.2 Load & build the device code
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

		AssignmentFunctions AF(context, queue, program);
		
		// declare initial vector
		std::vector<mytype> initialVec;
		std::vector<mytypef> initialVecf;

		// generic tokeniser + populating initial vector
		char delimiter = ' ';
		std::vector<string> splitText(0);
		std::ifstream txtFile("../../temp_lincolnshire_datasets/temp_lincolnshire.txt");
		string text;
		size_t start;
		size_t end;

		int e = 0;
		while (getline(txtFile, text)) {
			start = 0;
			end = 0;
			splitText.clear();
			while (end < text.length() && start < text.length()) {
				end = text.find(delimiter, start);

				if (end == string::npos)
					end = text.length();
				string tok = text.substr(start, end - start);
				if (!tok.empty())
					splitText.push_back(tok);
				start = end + 1;
			}
			initialVecf.push_back(stof(splitText[5]));
			e++;
			if (!(e % 100000))
				cout << e << " Records Loaded \n";
		}
		txtFile.close();
		cout << "\n \n \n \nFile Loaded. \n";
		// Test Vector
		// std::vector<mytypef> testVecf{1.0, 3.0, 1.0};

		// 
		int initial_size = initialVecf.size();
		int local_size = 512;

		// get average
		std::vector<mytypef> Total = AF.callReduceFunctionFloat(initialVecf, "reduce_add_assignment_float", local_size);
		float averageVal = Total[0] / initial_size;

		std::vector<mytypef> sigmoidComponentF = AF.callMapFunctionFloat(initialVecf, "map_sd_assignment_float", local_size, averageVal, 0.001);
		
		std::vector<mytypef> sigmoidTotal = AF.callReduceFunctionFloat(sigmoidComponentF, "reduce_add_assignment_float", local_size);
		
		std::vector<mytypef> minInit = AF.callReduceFunctionFloat(initialVecf, "reduce_minimum_assignment_float", local_size);
		
		std::vector<mytypef> maxInit = AF.callReduceFunctionFloat(initialVecf, "reduce_maximum_assignment_float", local_size);


		//ATTEMPT SORT
		//std::vector<mytypef> sorted = AF.callSortFunctionFloat(initialVecf, "oddeven_assignment_float", local_size);
		//int sortedSize = sorted.size();
		//std::cout << "Median = " << sorted[round(sortedSize * 0.5)] << std::endl;
		//std::cout << "Upper Quartile = " << sorted[(sorted.size() / 4) * 3] << std::endl;
		//std::cout << "Lower Quartile = " << sorted[sorted.size() / 4] << std::endl;




		std::cout << "AverageVal = " << averageVal << std::endl;
		std::cout << "Initial Size = " << initial_size << std::endl;
		std::cout << "MinVal = " << minInit << std::endl;
		std::cout << "MaxVal = " << maxInit << std::endl;
		float variance = sigmoidTotal[0] / initial_size;
		float sd = sqrt(variance);
		std::cout << "S.D = " << sd << std::endl;
	}

	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}