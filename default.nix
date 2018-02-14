with import <nixpkgs> {};
stdenv.mkDerivation {
  name = "env";
  buildInputs = [
    bashInteractive
    python3
  ] ++ (with python36Packages; [
    jupyter
    numpy
    matplotlib
    pandas
  ]);
}
