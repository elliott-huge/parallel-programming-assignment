
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
		//2.1 Select computing devices
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
		//Part 3 - memory allocation
		//host - input
		//std::vector<mytype> A(1800000, 5);
		// import le data
		
		// declare initial vector
		std::vector<mytype> initialVec;
		std::vector<mytypef> initialVecf;

		// generic tokeniser and populating initial vector
		char delimiter = ' ';
		std::vector<string> splitText(0);
		std::ifstream txtFile("../../temp_lincolnshire_datasets/temp_lincolnshire.txt");
		string text;
		size_t start;
		size_t end;
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
		}
		txtFile.close();
		
		//cout << initialVecf << endl;
		// testing occurs here
		//  1.1,2.8,3.9,1.1,1.4,1.5,6.3,8.2,5.999,12.2,0.0,0.1 
		std::vector<mytypef> testVecf{1.0, 3.0, 1.0};
		int initial_size = initialVecf.size();
		int local_size = 256;

		// get average
		// this will work regardless of temp values being 0.0
		std::vector<mytypef> Total = AF.callReduceFunctionFloat(initialVecf, "reduce_add_assignment_float", local_size);
		float averageVal = Total[0] / initial_size;
		std::cout << "Total = " << Total << std::endl;
		std::cout << "AverageVal = " << averageVal << std::endl;

		// get sigmoid
		// 0.0 INCOMPATIBLE
		// TODO make work for float
		//std::vector<mytype> sigmoidComponent = AF.callMapFunction(initialVec, "map_sd_assignment", local_size, averageVal);

		// 0.0 version that also works for floats
		std::vector<mytypef> sigmoidComponentF = AF.callMapFunctionFloat(initialVecf, "map_sd_assignment_float", local_size, averageVal, 0.001);



		// may produce a value too large to be stored lol
		// 0.0 compatable
		// TODO make work for float *d
		std::vector<mytypef> sigmoidTotal = AF.callReduceFunctionFloat(sigmoidComponentF, "reduce_add_assignment_float", local_size);

		// 0.0 compatable
		// TODO make work for float *d
		std::vector<mytypef> minInit = AF.callReduceFunctionFloat(initialVecf, "reduce_minimum_assignment_float", local_size);
		// 0.0 compatable
		// TODO make work for float *d
		std::vector<mytypef> maxInit = AF.callReduceFunctionFloat(initialVecf, "reduce_maximum_assignment_float", local_size);


		//ATTEMPT SORT
		std::vector<mytypef> sorted = AF.callSortFunctionFloat(initialVecf, "bubblesort_assignment_float", local_size);
		int sortedSize = sorted.size();
		std::cout << "Sorted Size = " << sortedSize << std::endl;
		//std::cout << "Median = " << sorted[sortedSize/2] << std::endl;
		//std::cout << "Upper Quartile = " << sorted[sorted.size() / 4] << std::endl;
		//std::cout << "Lower Quartile = " << sorted[sorted.size() / 4] << std::endl;
		std::cout << "MinVal = " << minInit << std::endl;
		std::cout << "MaxVal = " << maxInit << std::endl;


		//std::cout << "AverageVal = " << averageVal << std::endl;
		//std::cout << "SigmoidTotal = " << sigmoidTotal << std::endl;
		float variance = sigmoidTotal[0] / initial_size;
		float sd = sqrt(variance);
		std::cout << "S.D = " << sd << std::endl;


		// TODO calculate standard deviation
		// square root total
		// divide by n
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}