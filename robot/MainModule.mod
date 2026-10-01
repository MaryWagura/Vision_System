MODULE MainModule
    !=====================================================================
    ! INA700 vision pick-and-place - ABB main module
    !
    ! Frames:  every move uses tool1234 \WObj:=wobj1234.
    ! Python must send pick X/Y in wobj1234 (marker values = aruco target).
    !
    ! Before running after ANY tool change:
    !   - re-measure PAPER_Z (cup sucked onto paper, read Z in wobj1234)
    !   - re-teach p20 / p30 / p40 (place positions)
    !   - first run in Manual reduced speed, stepping one instruction
    !=====================================================================

    TASK PERS wobjdata wobj1234:=[FALSE,TRUE,"",[[478.279,-114.169,-10.3565],[0.999956,-0.00175295,0.000717086,0.00922622]],[[0,0,0],[1,0,0,0]]];
    TASK PERS tooldata tool1234:=[TRUE,[[-96.07,0.34481,181.882],[1,0,0,0]],[0.5,[50,0,50],[1,0,0,0],0,0,0]];

    ! Place positions on the designated-positions board (wobj1234)
    CONST robtarget p20:=[[273.73,-66.71,14.75],[0.0110135,-0.0258492,-0.999605,0.000494662],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget p30:=[[205.88,-136.26,15.07],[0.0110339,-0.0258024,-0.999606,0.000509208],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget p40:=[[261.03,-195.60,14.50],[0.0110429,-0.0257536,-0.999607,0.000508994],[-1,-1,-1,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];

    ! Taught on the ArUco marker centre (wobj1234). Its X/Y must match
    ! MARKER_ROBOT_X_MM / MARKER_ROBOT_Y_MM in the Python script.
    CONST robtarget aruco:=[[115.38,208.96,5.56],[0.0324101,0.0123748,-0.999344,-0.0103724],[0,0,0,0],[9E+09,9E+09,9E+09,9E+09,9E+09,9E+09]];

    ! The old p10 had Z = -53.92 (below the table) and caused the crash.
    ! Home is now derived from the aruco target: HOME_DZ mm above the marker.
    ! If you re-teach a proper home, you can replace pHome with it.
    VAR robtarget pHome;

    ! --- Heights (mm, wobj1234) ---
    CONST num PAPER_Z := 0;         ! measured: cup sucked onto the paper
    CONST num SHAPE_THICK := 10;    ! all shapes are 1 cm thick
    CONST num SQUEEZE := 0;         ! already included in PAPER_Z measurement
    CONST num MIN_Z := 2;           ! crash guard: never closer to the paper
    CONST num APPROACH_DZ := 80;    ! hover height above pick/place before descending
    CONST num HOME_DZ := 150;       ! home height above the marker
    CONST num PLACE_DZ := 0;        ! raise to ~5 if p20-p40 are not re-taught with the new tool

    ! --- Allowed pick area, relative to the marker centre (mm) ---
    ! Adjust to your paper. Targets outside are skipped, never moved to.
    CONST num REACH_X := 200;
    CONST num REACH_Y := 140;

    ! Set FALSE to skip the hover-over-marker calibration check
    CONST bool SHOW_ARUCO := TRUE;

    PROC PickAndPlace(robtarget pickTarget, robtarget placeTarget)
        MoveJ pHome, v500, fine, tool1234\WObj:=wobj1234;

        IF SHOW_ARUCO THEN
            ! Visual check: cup should hover exactly over the marker centre
            MoveJ Offs(aruco, 0, 0, APPROACH_DZ), v200, fine, tool1234\WObj:=wobj1234;
        ENDIF

        TPWrite "Moving to pick object";
        MoveJ Offs(pickTarget, 0, 0, APPROACH_DZ), v200, fine, tool1234\WObj:=wobj1234;
        MoveL pickTarget, v50, fine, tool1234\WObj:=wobj1234;

        TPWrite "Vacuum ON";
        Set doValve1;
        WaitTime 0.5;
        MoveL Offs(pickTarget, 0, 0, APPROACH_DZ), v100, fine, tool1234\WObj:=wobj1234;

        MoveJ pHome, v500, fine, tool1234\WObj:=wobj1234;

        TPWrite "Moving to placing position";
        MoveJ Offs(placeTarget, 0, 0, APPROACH_DZ), v200, fine, tool1234\WObj:=wobj1234;
        MoveL Offs(placeTarget, 0, 0, PLACE_DZ), v50, fine, tool1234\WObj:=wobj1234;

        TPWrite "Vacuum OFF";
        Reset doValve1;
        WaitTime 1;
        MoveL Offs(placeTarget, 0, 0, APPROACH_DZ), v100, fine, tool1234\WObj:=wobj1234;

        MoveJ pHome, v500, fine, tool1234\WObj:=wobj1234;
    ENDPROC

    PROC main()
        VAR string visionData;
        VAR num comma1;
        VAR num comma2;
        VAR num comma3;
        VAR string str_shape;
        VAR string str_x;
        VAR string str_y;
        VAR string str_angle;
        VAR num val_x;
        VAR num val_y;
        VAR num val_angle;
        VAR bool ok_x;
        VAR bool ok_y;
        VAR bool ok_angle;
        VAR num pickZ;
        VAR bool known_shape;
        VAR robtarget dynamicPickTarget;
        VAR robtarget staticPlaceTarget;

        pHome := Offs(aruco, 0, 0, HOME_DZ);

        TPWrite "Vision Pick and Place Started";

        RobotAsClientConnect;

        WHILE TRUE DO
            TPWrite "Requesting targets from camera...";
            RobotClienSendMessage("REQUEST_COORDS");

            visionData := RobotClientReciveMessage();

            IF visionData = "" OR visionData = "NO_TARGET,0,0,0" THEN
                TPWrite "Camera sees no shapes. Waiting 2 seconds...";
                WaitTime 2;
            ELSE
                comma1 := StrFind(visionData, 1, ",");

                IF comma1 > 0 AND comma1 < StrLen(visionData) THEN
                    comma2 := StrFind(visionData, comma1 + 1, ",");
                ELSE
                    comma2 := 0;
                ENDIF

                IF comma2 > 0 AND comma2 < StrLen(visionData) THEN
                    comma3 := StrFind(visionData, comma2 + 1, ",");
                ELSE
                    comma3 := 0;
                ENDIF

                IF comma1 > 0 AND comma2 > 0 AND comma3 > 0 THEN
                    str_shape := StrPart(visionData, 1, comma1 - 1);
                    str_x := StrPart(visionData, comma1 + 1, comma2 - comma1 - 1);
                    str_y := StrPart(visionData, comma2 + 1, comma3 - comma2 - 1);
                    str_angle := StrPart(visionData, comma3 + 1, StrLen(visionData) - comma3);

                    ok_x := StrToVal(str_x, val_x);
                    ok_y := StrToVal(str_y, val_y);
                    ok_angle := StrToVal(str_angle, val_angle);
                    IF NOT ok_angle THEN
                        val_angle := 0;
                    ENDIF

                    TPWrite "Detected: " + str_shape;

                    ! Python sends lowercase names: circle / star / square
                    known_shape := TRUE;
                    IF str_shape = "circle" THEN
                        staticPlaceTarget := p20;
                    ELSEIF str_shape = "star" THEN
                        staticPlaceTarget := p30;
                    ELSEIF str_shape = "square" THEN
                        staticPlaceTarget := p40;
                    ELSE
                        known_shape := FALSE;
                    ENDIF

                    ! Pick height: top face of a 1 cm shape
                    pickZ := PAPER_Z + SHAPE_THICK - SQUEEZE;
                    IF pickZ < MIN_Z THEN
                        pickZ := MIN_Z;
                    ENDIF

                    IF NOT ok_x OR NOT ok_y THEN
                        TPWrite "Bad coordinates received: " + visionData;
                        WaitTime 2;
                    ELSEIF NOT known_shape THEN
                        TPWrite "Unknown shape - skipped: " + str_shape;
                        WaitTime 2;
                    ELSEIF Abs(val_x - aruco.trans.x) > REACH_X OR Abs(val_y - aruco.trans.y) > REACH_Y THEN
                        TPWrite "Target outside pick area - skipped";
                        WaitTime 2;
                    ELSE
                        ! Base the pick on the aruco target: taught on the paper,
                        ! tool pointing down, arm configuration for the paper area
                        dynamicPickTarget := aruco;
                        dynamicPickTarget.trans := [val_x, val_y, pickZ];
                        dynamicPickTarget := RelTool(dynamicPickTarget, 0, 0, 0 \Rz:=val_angle);

                        TPWrite "Press yellow button to pick " + str_shape;
                        WaitDI diOkButton, 1;

                        PickAndPlace dynamicPickTarget, staticPlaceTarget;
                        TPWrite "Pick and place complete";
                    ENDIF
                ELSE
                    TPWrite "Bad data received: " + visionData;
                    WaitTime 2;
                ENDIF
            ENDIF
        ENDWHILE
    ENDPROC
ENDMODULE
