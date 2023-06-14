var xLeft = 40;
var yTop = 130;
var roundCornerBlock = 5;
var roundCornerBus = 2;
var widthBlock = 125;
var widthBus = 10;
var heightBlock = 60;
var heightBus = 6 * heightBlock;
var lengthFlow = 90;

var lineCorrectionWidthBlock = 1;
var lineCorrectionLengthFlow = 1;

/************************************************************/
/*                ENABLING DISABLING OPTIONS                */
/************************************************************/

function check_optimization_strategy(id) {
    // Update the styles after changing the optimization strategy
    styleBlock(id);
    styleText(id);
    styleLine(id);
    styleArrow(id);
    styleInformation(id);

    if (document.getElementById(id+"Design").checked) {
        document.getElementById(id+"NominalCapacity").disabled = true; 
        document.getElementById("lbl"+toTitleCase(id)+"NominalCapacity").classList.add('disabled');
        document.getElementById(id+"NominalCapacityUnit").classList.add('disabled');
    } else {
        document.getElementById(id+"NominalCapacity").disabled = false; 
        document.getElementById("lbl"+toTitleCase(id)+"NominalCapacity").classList.remove('disabled');
        document.getElementById(id+"NominalCapacityUnit").classList.remove('disabled');
    }
}
function check_box_visibility(id) {
    if (id === 'inverter' && !document.getElementById('selectInverter').checked) {
        // If it's not checked, then uncheck 'selectPv' and 'selectBattery'
        document.getElementById('selectPv').checked = false;
        document.getElementById('selectBattery').checked = false;
        change_box_visibility(id);
        let list = ['pv', 'battery'];
        for (let i = 0; i < list.length; i++) {
            change_box_visibility(list[i]);
            refreshBlocksOnDiagram(list[i]);
        }
    }
    if (id === 'pv' && document.getElementById('selectPv').checked) {
        document.getElementById('selectInverter').checked = true;
        change_box_visibility('inverter');
        refreshBlocksOnDiagram('inverter');
    }
    if (id === 'battery' && document.getElementById('selectBattery').checked) {
        document.getElementById('selectInverter').checked = true;
        change_box_visibility('inverter');
        refreshBlocksOnDiagram('inverter');
    }
    if (id === 'battery'
        && !document.getElementById('selectBattery').checked
        && !document.getElementById('selectPv').checked) {
        document.getElementById('selectInverter').checked = false;
        change_box_visibility('inverter');
        refreshBlocksOnDiagram('inverter');
    }
    if (id === 'pv'
        && !document.getElementById('selectBattery').checked
        && !document.getElementById('selectPv').checked) {
        document.getElementById('selectInverter').checked = false;
        change_box_visibility('inverter');
        refreshBlocksOnDiagram('inverter');
    }
    change_box_visibility(id);
}


function change_box_visibility(id) {
    // Depending on the selection of each box (i.e., component), the visibility
    // of the box components will be changed
    var component_specifications = {
        'dieselGenset': [
            'Design', 'Dispatch', 
            'NominalCapacity', 'Lifetime', 'Capex', 'Opex',
            'VariableCost', 'FuelCost', 'FuelLhv', 
            'MinEfficiency', 'MaxEfficiency', 'MinLoad', 'MaxLoad'
        ],
        'battery': [
            'Design', 'Dispatch', 
            'NominalCapacity', 'Lifetime', 'Capex', 'Opex',
            'SocMin', 'SocMax', 'CrateIn', 'CrateOut', 'Efficiency'
        ],
        'pv': [
            'Design', 'Dispatch',
            'NominalCapacity', 'Lifetime', 'Capex', 'Opex', 
            'FileImportSolarPotential',
        ],
        'inverter': [
            'Design', 'Dispatch',
            'NominalCapacity', 'Lifetime', 'Capex', 'Opex', 'Efficiency',
        ],
        'rectifier': [
            'Design', 'Dispatch',
            'NominalCapacity', 'Lifetime', 'Capex', 'Opex', 'Efficiency',
        ],
        'shortage':[
            'MaxTotal', 'MaxTimestep', 'PenaltyCost'
        ]
    };
    
    // ---------- ENABLING ITEMS ---------- //
    if (document.getElementById("select"+toTitleCase(id)).checked) {
        // Change the border color.
        // document.getElementById("select"+toTitleCase(id)+"Box").style.border = '2px solid #198754';
        document.getElementById("select"+toTitleCase(id)+"Box").classList.remove('box--not-selected');

        for (index in component_specifications[id]) {
            // First, get the property listed in the above dictionary.
            property = component_specifications[id][index]
            
            // All fields as well as the `design` and `dispatch` buttons.
            document.getElementById(id+property).disabled = false;
            
            // All labels.
            if (document.getElementById("lbl"+toTitleCase(id)+property)) {
                document.getElementById("lbl"+toTitleCase(id)+property).classList.remove('disabled');
            }
            
            // All units and in case of PV, the button for impoting solar potential.
            if (document.getElementById(id+property+"Unit")) {
                document.getElementById(id+property+"Unit").classList.remove('disabled');
            } else if (document.getElementById("btn"+toTitleCase(id)+property)) {
                document.getElementById("btn"+toTitleCase(id)+property).classList.remove('disabled');
            }

            // Check the optimization mode for enabling/disabling capacity.
            if (id != "shortage") {
                check_optimization_strategy(id);
            }
        }
    // ---------- DISABLING ITEMS ---------- //
    } else {
        // Change the border color.
        // document.getElementById("select"+toTitleCase(id)+"Box").style.border = '2px solid #dc3545';  
        document.getElementById("select"+toTitleCase(id)+"Box").classList.add('box--not-selected');

        for (index in component_specifications[id]) {
            // First, get the property listed in the above dictionary.
            property = component_specifications[id][index]

            // All fields as well as the `diesgn` and `dispatch` buttons.
            document.getElementById(id+property).disabled = true;

            // All labels.
            if (document.getElementById("lbl"+toTitleCase(id)+property)) {
                document.getElementById("lbl"+toTitleCase(id)+property).classList.add('disabled');
            }
            
            // All units and in case of PV, the button for impoting solar potential.
            if (document.getElementById(id+property+"Unit")) {
                document.getElementById(id+property+"Unit").classList.add('disabled');
            } else if (document.getElementById("btn"+toTitleCase(id)+property)) {
                document.getElementById("btn"+toTitleCase(id)+property).classList.add('disabled');
            }
        }
   }
}


function refreshBlocksOnDiagramOnLoad(){
    const component = [
        'pv', 
        'battery', 
        'dieselGenset', 
        'inverter', 
        'rectifier',
        'shortage',
        // 'surplus',
    ];

    for (let i=0; i<component.length; i++) {
        refreshBlocksOnDiagram(component[i]);
        if (component[i] !== 'shortage'){
        // if (component[i] !== 'shortage' && component [i] !== 'surplus'){
            check_box_visibility(component[i]);
            check_optimization_strategy(component[i]);
        };
    }
    refreshBlocksOnDiagram('demand');
}


/************************************************************/
/*                 DRAW AND STYLE THE BLOCKS                */
/************************************************************/
function drawBlock(id, x, y) {
    const block = document.getElementById("block"+toTitleCase(id));

    if (id.slice(2, 6) === "Bus") {
        rxy = roundCornerBus;
        width = widthBus;
        height = heightBus;
    } else {
        rxy = roundCornerBlock;
        width = widthBlock;
        height = heightBlock;
    }

    block.setAttribute('x', x);
    block.setAttribute('y', y);
    block.setAttribute('rx', rxy);
    block.setAttribute('ry', rxy);
    block.setAttribute('width', width);
    block.setAttribute('height', height);
}

function styleBlock(id) {
    const block = document.getElementById("block"+toTitleCase(id));
    if (id === 'demand') {
        block.classList.add('components-block--demand');
    } else if (id === 'shortage') {
    // } else if (id === 'shortage' || id === 'surplus') {
        block.classList.add('components-block--constraints');
    } else if (document.getElementById(id+"Design").checked) {
        block.classList.remove('components-block--dispatch');
        block.classList.add('components-block--design');
    } else {
        block.classList.remove('components-block--design');
        block.classList.add('components-block--dispatch');
    }
}


/************************************************************/
/*                    WRITE THE BLOCK TEXT                  */
/************************************************************/
function writeText(id, x, y) {
    const text = document.getElementById("text"+toTitleCase(id));

    text.setAttribute('x', x);
    text.setAttribute('y', y);
}

function writeInformation(id, x, y) {
    const information = document.getElementById("information"+toTitleCase(id));
    
    if (id !== 'demand') {
        information.setAttribute('x', x);
        information.setAttribute('y', y);
        if (id == 'shortage'){
            const informationSecondLine = document.getElementById("information"+toTitleCase(id)+"SecondLine");
            informationSecondLine.setAttribute('x', x);
            informationSecondLine.setAttribute('y', 0.9*y);
        }
    }
}

function styleText(id) {
    const text = document.getElementById("text"+toTitleCase(id));

    if (id === 'demand') {
        text.classList.add('components-text--demand');
    } else if (id === 'shortage') {
    // } else if (id === 'shortage' || id === 'surplus') {
        text.classList.add('components-text--constraints');
    } else if (document.getElementById(id+"Design").checked) {
        text.classList.remove('components-text--dispatch');
        text.classList.add('components-text--design');
    } else {
        text.classList.remove('components-text--design');
        text.classList.add('components-text--dispatch');
    }

}

function styleInformation(id) {
    const information = document.getElementById("information"+toTitleCase(id));
    if (id === 'demand') {
        
    } else if (id === 'shortage') {
        // } else if (id === 'shortage' || id === 'surplus') {
        const informationSecondLine = document.getElementById("information"+toTitleCase(id)+"SecondLine");
        const percentageTotal = document.getElementById(id+"MaxTotal").value;
        const percentageTimestep = document.getElementById(id+"MaxTimestep").value;
        const unit = document.getElementById(id+"MaxTotalUnit").innerText;
        information.textContent="max. each timestep " + percentageTimestep + unit;
        informationSecondLine.textContent="max. total " + percentageTotal + unit;
        information.classList.add('components-information--constraints');
        informationSecondLine.classList.add('components-information--constraints');
    } else if (document.getElementById(id+"Design").checked) {
        information.textContent="design";
        information.classList.remove('components-information--dispatch');
        information.classList.add('components-information--design');
    } else {
        const capacity = document.getElementById(id+"NominalCapacity").value;
        const unit = document.getElementById(id+"NominalCapacityUnit").innerText;
        information.textContent="dispatch - " + capacity + " " + unit;
        information.classList.remove('components-information--design');
        information.classList.add('components-information--dispatch');
    }

}

function drawLine(id, linePoints1, linePoints2) {
    // Lines always start from one side of the blocks and end at the bus

    // id is in form of for example linePV or lineDieselGenset
    const line1 = document.getElementById("line"+toTitleCase(id));

    line1.setAttribute('x1', linePoints1[0][0]);
    line1.setAttribute('y1', linePoints1[0][1]);
    line1.setAttribute('x2', linePoints1[1][0]);
    line1.setAttribute('y2', linePoints1[1][1]);

    // For inverter and rectifier there should be two lines.
    if (linePoints2.length > 0) {
        const line2 = document.getElementById("line"+toTitleCase(id)+"2");

        line2.setAttribute('x1', linePoints2[0][0]);
        line2.setAttribute('y1', linePoints2[0][1]);
        line2.setAttribute('x2', linePoints2[1][0]);
        line2.setAttribute('y2', linePoints2[1][1]);
    }
}

function styleLine(id){
    const line1 = document.getElementById("line"+toTitleCase(id));

    if (id === 'demand') {
        line1.classList.add('components-flow--demand');
    } else if (id === 'shortage') {
    // } else if (id === 'shortage' || id === 'surplus') {
        line1.classList.add('components-flow--constraints');
    } else if (document.getElementById(id+"Design").checked) {
        line1.classList.remove('components-flow--dispatch');
        line1.classList.add('components-flow--design');
    } else {
        line1.classList.remove('components-flow--design');
        line1.classList.add('components-flow--dispatch');
    }

    // For inverter and rectifier there should be two lines.
    if (id === "inverter" || id === "rectifier") {
        const line2 = document.getElementById("line"+toTitleCase(id)+"2");

        if (document.getElementById(id+"Design").checked) {
            line2.classList.remove('components-flow--dispatch');
            line2.classList.add('components-flow--design');
            
        } else {
            line2.classList.remove('components-flow--design');
            line2.classList.add('components-flow--dispatch');
        }
    }
    
}

function drawArrow(id, arrowOutPoints1, arrowInPoints1, arrowOutPoints2, arrowInPoints2) {
    // The default arrow is the `arrowOut` which always at the end of the line, 
    // that means it is outward (block ---> bus ).
    // Another type of arrow is called `arrowIn`, which corresponds to the arrows
    // entering a block (bus ---> block).

    // points should be in the format [[x1,y1], [x2,y2], [x3,y3]]
    const arrowOut1 = document.getElementById("arrowOut"+toTitleCase(id));
    const arrowIn1 = document.getElementById("arrowIn"+toTitleCase(id));

    arrowOut1.setAttribute('points', arrowOutPoints1);
    arrowIn1.setAttribute('points', arrowInPoints1);

    // For inverter and rectifier there are two lines and therefore, two arrows are required
    if (arrowOutPoints2.length > 0) {
        const arrowOut2 = document.getElementById("arrowOut"+toTitleCase(id)+"2");
        const arrowIn2 = document.getElementById("arrowIn"+toTitleCase(id)+"2");

        arrowOut2.setAttribute('points', arrowOutPoints2);
        arrowIn2.setAttribute('points', arrowInPoints2);
    }
}

function styleArrow(id){
    const arrowOut1 = document.getElementById("arrowOut"+toTitleCase(id));
    const arrowIn1 = document.getElementById("arrowIn"+toTitleCase(id));

    if (id === 'demand' ) {        
        $(arrowOut1).attr("visibility", "hidden");
        arrowIn1.classList.add('components-flow--demand');
    } else if (id === 'shortage') {        
        $(arrowIn1).attr("visibility", "hidden");
        arrowOut1.classList.add('components-flow--constraints');
    // } else if (id === 'surplus') {        
    //     $(arrowOut1).attr("visibility", "hidden");
    //     arrowIn1.classList.add('components-flow--constraints');
    } else if (document.getElementById(id+"Design").checked) {
        if (id === 'pv' || id === 'dieselGenset' || id === 'shortage'){
            $(arrowIn1).attr("visibility", "hidden");
            arrowOut1.classList.remove('components-flow--dispatch');
            arrowOut1.classList.add('components-flow--design');        
        } else if (id === 'battery'){
            arrowOut1.classList.remove('components-flow--dispatch');
            arrowIn1.classList.remove('components-flow--dispatch');
            arrowOut1.classList.add('components-flow--design');        
            arrowIn1.classList.add('components-flow--design');        
        } else {
            const arrowOut2 = document.getElementById("arrowOut"+toTitleCase(id)+"2");
            const arrowIn2 = document.getElementById("arrowIn"+toTitleCase(id)+"2");
            if (id === 'rectifier') {
                $(arrowOut1).attr("visibility", "hidden");
                $(arrowIn2).attr("visibility", "hidden");
                arrowOut2.classList.remove('components-flow--dispatch');
                arrowIn1.classList.remove('components-flow--dispatch');
                arrowOut2.classList.add('components-flow--design');        
                arrowIn1.classList.add('components-flow--design');        
            } else {
                $(arrowOut2).attr("visibility", "hidden");
                $(arrowIn1).attr("visibility", "hidden");
                arrowOut1.classList.remove('components-flow--dispatch');
                arrowIn2.classList.remove('components-flow--dispatch');
                arrowOut1.classList.add('components-flow--design');        
                arrowIn2.classList.add('components-flow--design');        
            }
        };
    } else {
        if (id === 'pv' || id === 'dieselGenset' || id === 'shortage'){
            $(arrowIn1).attr("visibility", "hidden");
            arrowOut1.classList.add('components-flow--dispatch');
            arrowOut1.classList.remove('components-flow--design');        
        } else if (id === 'battery'){
            arrowOut1.classList.add('components-flow--dispatch');
            arrowIn1.classList.add('components-flow--dispatch');
            arrowOut1.classList.remove('components-flow--design');        
            arrowIn1.classList.remove('components-flow--design');        
        } else {
            const arrowOut2 = document.getElementById("arrowOut"+toTitleCase(id)+"2");
            const arrowIn2 = document.getElementById("arrowIn"+toTitleCase(id)+"2");
            if (id === 'inverter') {
                $(arrowOut1).attr("visibility", "hidden");
                $(arrowIn2).attr("visibility", "hidden");
                arrowOut2.classList.add('components-flow--dispatch');
                arrowIn1.classList.add('components-flow--dispatch');
                arrowOut2.classList.remove('components-flow--design');        
                arrowIn1.classList.remove('components-flow--design');        
            } else {
                $(arrowOut2).attr("visibility", "hidden");
                $(arrowIn1).attr("visibility", "hidden");
                arrowOut1.classList.add('components-flow--dispatch');
                arrowIn2.classList.add('components-flow--dispatch');
                arrowOut1.classList.remove('components-flow--design');        
                arrowIn2.classList.remove('components-flow--design');        
            }
        };    }

    // For inverter and rectifier there should be two lines.
    if (id === "inverter" || id === "rectifier") {
        const line2 = document.getElementById("line"+toTitleCase(id)+"2");

        if (document.getElementById(id+"Design").checked) {
            line2.classList.remove('components-flow--dispatch');
            line2.classList.add('components-flow--design');
            
        } else {
            line2.classList.remove('components-flow--design');
            line2.classList.add('components-flow--dispatch');
        }
    }
    
}

function refreshBusesOnDiagram(){
    // This function draw/remove AC and DC buses and their texts in the diagram 
    // depending on if the attached blocks to them are selected or not. 
    const groupDcBus = document.getElementById("groupDcBus");
    const groupAcBus = document.getElementById("groupAcBus");

    var busCoordinates = {
        'dcBus': {
            'x': xLeft + widthBlock + lengthFlow,
            'y': yTop - heightBlock,
        },
        'acBus': {
            'x': xLeft + 2 * widthBlock + 3 * lengthFlow + widthBus,
            'y': yTop - heightBlock,
        },
    };

    const selectPv = document.getElementById("selectPv").checked;
    const selectBattery = document.getElementById("selectBattery").checked;
    const selectInverter = document.getElementById("selectInverter").checked;
    const selectRectifier = document.getElementById("selectRectifier").checked;

    // Since there is always demand, AC bus is always visible
    $(groupAcBus).attr("visibility", "visible");
    drawBlock(
        id="acBus", 
        x=busCoordinates.acBus.x,
        y=busCoordinates.acBus.y,
    )
    writeText(
        id="acBus", 
        x=busCoordinates.acBus.x + 0.5 * widthBus,
        y=0.7*busCoordinates.acBus.y
    )

    // DC bus is not necessarily always visible
    if (selectPv || selectBattery || selectInverter || selectRectifier) {
        $(groupDcBus).attr("visibility", "visible");
        drawBlock(
            id="dcBus", 
            x=busCoordinates.dcBus.x,
            y=busCoordinates.dcBus.y,
        )
        writeText(
            id="dcBus", 
            x=busCoordinates.dcBus.x + 0.5 * widthBus,
            y=0.7*busCoordinates.dcBus.y
            )
    } else {
        // First make the SVG group visible
        $(groupDcBus).attr("visibility", "hidden");
    }

}

function refreshBlocksOnDiagram(id){
    // This function draw/remove all blocks and their texts and flows in the diagram depending on
    // if they are selected by user or not. 
    // For AC and DC buses, the function `refreshBusesOnDiagram` does the same work.
    const groupId = document.getElementById("group"+toTitleCase(id));

    if (id === 'demand') {
        var isSelected = true;
    } else if (id === 'shortage') {
    // } else if (id === 'shortage' || id === 'surplus') {
        if (document.getElementById("selectShortage").checked){
            var isSelected = document.getElementById("select"+toTitleCase(id)).checked;
        } else {
            var isSelected = false;
        }
    } else {
        var isSelected = document.getElementById("select"+toTitleCase(id)).checked;
    }

    var blockCoordinates = {
        'pv': {
            'x': xLeft,
            'y': yTop,
        },
        'battery': {
            'x': xLeft,
            'y': yTop + 3 * heightBlock,
        },
        'inverter': {
            'x': xLeft + widthBlock + 2 * lengthFlow + widthBus,
            'y': yTop - 0.5 * heightBlock,
        },
        'rectifier': {
            'x': xLeft + widthBlock + 2 * lengthFlow + widthBus,
            'y': yTop - 0.5 * heightBlock + 2 * heightBlock,
        },
        'dieselGenset': {
            'x': xLeft + widthBlock + 2 * lengthFlow + widthBus,
            'y': yTop - heightBlock / 2 + 4 * heightBlock,
        },
        'shortage': {
            'x': xLeft+ 2 * widthBlock + 4 * lengthFlow + 2 * widthBus,
            'y': yTop + 0.5 * heightBlock,
        },
        'demand': {
            'x': xLeft+ 2 * widthBlock + 4 * lengthFlow + 2 * widthBus,
            'y': yTop - heightBlock + 3.5 * heightBlock,
        },
        'surplus': {
            'x': xLeft+ 2 * widthBlock + 4 * lengthFlow + 2 * widthBus,
            'y': yTop - heightBlock + 5 * heightBlock,
        },
    };

    if (isSelected) {
        // First make the SVG group visible
        $(groupId).attr("visibility", "visible");

        /**************/
        /*   BLOCKS   */
        /**************/
        drawBlock(
            id=id, 
            x=blockCoordinates[id].x,
            y=blockCoordinates[id].y,
        )
        styleBlock(id=id);
        
        /*************/
        /*   TEXTS   */
        /*************/
        writeText(
            id=id,
            x=blockCoordinates[id].x + 0.5 * widthBlock,
            y=blockCoordinates[id].y + 0.5 * heightBlock
        )
        styleText(id);
        
        writeInformation(
            id=id,
            x=blockCoordinates[id].x,
            y=blockCoordinates[id].y - 0.1 * heightBlock,
        );
        styleInformation(id);

        
        /***********************/
        /*   LINES AND ARROWS  */
        /***********************/
        if (id === 'demand' || id === 'shortage') {
        // if (id === 'demand' || id === 'surplus' || id === 'shortage') {
            lineCorrectionWidthBlock = 0;
            lineCorrectionLengthFlow = -1;
        } else {
            lineCorrectionWidthBlock = 1;
            lineCorrectionLengthFlow = 1;
        };    
        linePoints1 = [
            [blockCoordinates[id].x + lineCorrectionWidthBlock * widthBlock, blockCoordinates[id].y + 0.5 * heightBlock],
            [blockCoordinates[id].x + lineCorrectionWidthBlock * widthBlock + lineCorrectionLengthFlow * lengthFlow, blockCoordinates[id].y + 0.5 * heightBlock]
        ];

        arrowOutPoints1 = [
            [
                linePoints1[1][0] - lineCorrectionLengthFlow * 0.15 * lengthFlow,
                linePoints1[1][1] - lineCorrectionLengthFlow * 0.1 * lengthFlow
            ],
            [linePoints1[1][0], linePoints1[1][1]],
            [
                linePoints1[1][0] - lineCorrectionLengthFlow * 0.15 * lengthFlow,
                linePoints1[1][1] + lineCorrectionLengthFlow * 0.1 * lengthFlow
            ],
        ];

        arrowInPoints1 = [
            [
                linePoints1[0][0] + lineCorrectionLengthFlow * 0.15 * lengthFlow,
                linePoints1[0][1] - lineCorrectionLengthFlow * 0.1 * lengthFlow
            ],
            [linePoints1[0][0], linePoints1[1][1]],
            [
                linePoints1[0][0] + lineCorrectionLengthFlow * 0.15 * lengthFlow,
                linePoints1[0][1] + lineCorrectionLengthFlow * 0.1 * lengthFlow
            ],
        ];

        // For inverter and rectifier there would be two lines
        if (id === "inverter" || id === "rectifier") {
            lineCorrectionWidthBlock = 0;
            lineCorrectionLengthFlow = -1;
            linePoints2 = [
                [blockCoordinates[id].x + lineCorrectionWidthBlock * widthBlock, blockCoordinates[id].y + 0.5 * heightBlock],
                [blockCoordinates[id].x + lineCorrectionWidthBlock * widthBlock + lineCorrectionLengthFlow * lengthFlow, blockCoordinates[id].y + 0.5 * heightBlock]
            ];

            arrowOutPoints2 = [
                [
                    linePoints2[1][0] - lineCorrectionLengthFlow * 0.15 * lengthFlow,
                    linePoints2[1][1] - lineCorrectionLengthFlow * 0.1 * lengthFlow
                ],
                [linePoints2[1][0], linePoints2[1][1]],
                [
                    linePoints2[1][0] - lineCorrectionLengthFlow * 0.15 * lengthFlow,
                    linePoints2[1][1] + lineCorrectionLengthFlow * 0.1 * lengthFlow
                ],
            ];
    
            arrowInPoints2 = [
                [
                    linePoints2[0][0] + lineCorrectionLengthFlow * 0.15 * lengthFlow,
                    linePoints2[0][1] - lineCorrectionLengthFlow * 0.1 * lengthFlow
                ],
                [linePoints2[0][0], linePoints2[1][1]],
                [
                    linePoints2[0][0] + lineCorrectionLengthFlow * 0.15 * lengthFlow,
                    linePoints2[0][1] + lineCorrectionLengthFlow * 0.1 * lengthFlow
                ],
            ];
        } else {
            linePoints2 = [];
            arrowOutPoints2 = [];
            arrowInPoints2 = [];
        }
        drawLine(
            id=id,
            linePoints1=linePoints1,
            linePoints2=linePoints2
        )
        styleLine(id);

        drawArrow(
            id=id,
            arrowOutPoints1=arrowOutPoints1,
            arrowInPoints1=arrowInPoints1,
            arrowOutPoints2=arrowOutPoints2,
            arrowInPoints2=arrowInPoints2,
        )
        styleArrow(id);

    } else {
        $(groupId).attr("visibility", "hidden");
    }

    refreshBusesOnDiagram();
}

function toTitleCase(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
}
