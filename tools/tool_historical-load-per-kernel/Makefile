release:
	$(CXX) -O3 -std=c++11 -shared -fPIC -o tool.so tool.cpp
	$(CXX) -O3 -std=c++11 -shared -fPIC -o tool_strat_only.so tool_strat_only.cpp
	$(CXX) -O3 -std=c++11 -shared -fPIC -o tool_stats_basic.so tool_stats_basic.cpp
	$(CXX) -O3 -std=c++11 -shared -fPIC -o tool_stats_per_kernel.so tool_stats_per_kernel.cpp

debug:
	$(CXX) -O0 -g -std=c++11 -shared -fPIC -o tool.so tool.cpp
	$(CXX) -O0 -g -std=c++11 -shared -fPIC -o tool_strat_only.so tool_strat_only.cpp
	$(CXX) -O0 -g -std=c++11 -shared -fPIC -o tool_stats_basic.so tool_stats_basic.cpp
	$(CXX) -O0 -g -std=c++11 -shared -fPIC -o tool_stats_per_kernel.so tool_stats_per_kernel.cpp
