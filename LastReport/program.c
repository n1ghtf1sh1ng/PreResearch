//
//  コマンドライン引数をすべて表示するプログラム
//
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>


int main(int argc, char **argv)
{
	int i;

	printf("argc=%d\n", argc);

	for (i = 0 ; i < argc ; i++)
	{
		printf("argv[%d] = \"%s\"\n", i, argv[i]);
	}

	return(0);
}
