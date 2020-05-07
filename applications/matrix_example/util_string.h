#ifndef __UTIL_STRING__
#define __UTIL_STRING__

#include <iostream>
#include <sstream>
#include <list>
#include <string>

std::list<std::string> split(const std::string& s, char delimiter)
{
    std::list<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}
#endif