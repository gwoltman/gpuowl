#!/bin/bash
cat <<EOM
// Copyright (C) Mihai Preda
// Generated file, do not edit. See genbundle.sh and src/cl/*.cl

#include <vector>

static const std::vector<const char*> CL_FILES{
EOM

names=

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

echo "static const std::vector<const char*> CL_FILE_NAMES{${names}};"

cat <<EOM
const std::vector<const char*>& getClFileNames() { return CL_FILE_NAMES; }
const std::vector<const char*>& getClFiles() { return CL_FILES; }
EOM
