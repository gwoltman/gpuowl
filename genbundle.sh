#!/bin/bash
# genbundle.sh <files of the main bundle> [-- <files of the gcd bundle>]
# The gcd bundle (src/cl/gcd/*.cl) is compiled into programs of its own: it is not part of the hash that keys the cached binaries
# of the main kernels (see KernelCompiler), so changing it does not recompile them.
cat <<EOM
// Copyright (C) Mihai Preda
// Generated file, do not edit. See genbundle.sh and src/cl/*.cl

#include <vector>

EOM

# emit <vector name> <files...>: the files' contents as a vector of C strings; their names are left in $names
emit() {
    local list=$1
    shift
    names=
    echo "static const std::vector<const char*> ${list}{"
    for xx in "$@"
    do
        x=$(basename "$xx")

        if [ "$x" = "genbundle.sh" ] ; then continue ; fi

        names=${names}\"${x}\",

        echo "// $xx"

        # MSVC cannot handle string constants longer than 16KB.  Thus, output one raw line at a time and concatenate them using the C++ preprocessor
        echo '        R"cltag('
        while IFS= read -r line; do
            echo ')cltag" R"cltag('"$line"
        done < $xx
        echo ')cltag",'

        echo
    done
    echo '};'
}

main=()
gcd=()
inGcd=0
for xx in "$@"
do
    if [ "$xx" = "--" ] ; then inGcd=1 ; continue ; fi
    if [ $inGcd = 1 ] ; then gcd+=("$xx") ; else main+=("$xx") ; fi
done

emit CL_FILES "${main[@]}"
echo "static const std::vector<const char*> CL_FILE_NAMES{${names}};"
echo
emit GCD_CL_FILES "${gcd[@]}"
echo "static const std::vector<const char*> GCD_CL_FILE_NAMES{${names}};"

cat <<EOM
const std::vector<const char*>& getClFileNames() { return CL_FILE_NAMES; }
const std::vector<const char*>& getClFiles() { return CL_FILES; }
const std::vector<const char*>& getGcdClFileNames() { return GCD_CL_FILE_NAMES; }
const std::vector<const char*>& getGcdClFiles() { return GCD_CL_FILES; }
EOM
