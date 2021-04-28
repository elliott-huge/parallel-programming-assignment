
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

		std::vector<mytype> initialVec{ 1,7,3,6,7,43,6,7,32,34,5,12312,34,21,34,456,90,2,2,2,2,1,3,4,5 };
		int initial_size = initialVec.size();
		int local_size = 256;

		std::vector<mytype> Total = AF.callReduceFunction(initialVec, "reduce_add_assignment", local_size);

		int averageVal = Total[0] / initial_size;

		std::cout << "Total = " << Total << std::endl;
		std::cout << "AverageVal = " << averageVal << std::endl;

		std::vector<mytype> sigmoidComponent = AF.callMapFunction(initialVec, "map_sd_assignment", local_size, averageVal);

		std::vector<mytype> sigmoidTotal = AF.callReduceFunction(sigmoidComponent, "reduce_add_assignment", local_size);

		std::vector<mytype> minInit = AF.callReduceFunction(initialVec, "reduce_minimum_assignment", local_size);

		std::vector<mytype> maxInit = AF.callReduceFunction(initialVec, "reduce_maximum_assignment", local_size);

		std::cout << "Min = " << minInit << std::endl;
		std::cout << "Max = " << maxInit << std::endl;


		//std::cout << "AverageVal = " << averageVal << std::endl;
		std::cout << "SigmoidTotal = " << sigmoidTotal << std::endl;
	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}

	return 0;
}